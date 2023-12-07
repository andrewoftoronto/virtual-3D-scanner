from typing import List, Tuple
from itertools import chain
import torch
import numpy as np
from .source_loader import RawSceneData

# List of groups where each frame in the group includes a pixel that should 
# occupy the same 3D spatial location.
# Each such frame entry is (name, Y pixel, X pixel). Name can be a substring 
# of the colour map file name.
'''depth_inference_groups = [
    [
        ('054', 1440 - 559, 2250),
        ('058', 1440 - 182, 1160)
    ],
    [
        ('069', 1440 - 752, 1418),
        ('066', 1440 - 743, 1390)
    ]
]'''
ground_truth_frames = ['066', '069']
orientation_constraints = [
    ['066', '067', '068', '069']
]
lineup_constraints = [
    ['066', '067', '068', '069']
]

def cloud_gen(scene: RawSceneData):
    ''' Generate a point cloud from the given raw scene data. '''

    znear = 0.0001
    zfar = 10000

    '''y_angle = 65 / 180 * np.pi
    scene.proj_params.fl_y = y_angle
    focal_length = scene.proj_params.h / (2 * np.tan(y_angle / 2))
    scene.proj_params.fl_x = 2 * np.arctan(scene.proj_params.w / (2 * focal_length))
    scene.proj_params.fl_x = np.float32(scene.proj_params.fl_x)
    scene.proj_params.fl_y = np.float32(scene.proj_params.fl_y)'''

    fl_x = scene.proj_params.fl_x
    fl_y = scene.proj_params.fl_y
    cx = scene.proj_params.cx
    cy = scene.proj_params.cy
    w = scene.proj_params.w
    h = scene.proj_params.h

    # Infer zparams using images that point to the same place.
    _infer_zparams(scene)

    inv = _inverse_projection_matrix(w, h, fl_x, fl_y, torch.tensor(scene.proj_params.znear), torch.tensor(scene.proj_params.zfar)).cuda().T

    coords = np.zeros([0,3], dtype=np.float64)
    colours = np.zeros([0,3], dtype=np.float64)
    normals = np.zeros([0,3], dtype=np.float64)
    for (i, frame) in enumerate(scene.frames):
        #onlys = ['54', '58']
        onlys = ['54', '58', '62', '66', '69', '85']
        if not any([only in frame.colour_path for only in onlys]):
            continue
        #if i >= 20:
        #    continue

        transform_matrix = torch.from_numpy(frame.transform).to(torch.float32).cuda()
        inv_view = transform_matrix

        depth_map = torch.from_numpy(frame.get_depth()).cuda()

        # TODO: Codify this better.
        colour_map = torch.from_numpy(frame.get_colour()).cuda()
        colour_map = torch.flip(colour_map, dims=[0])
        colour_map = colour_map[:,:,:3] / 255

        full_depths = scene.proj_params.znear + (scene.proj_params.zfar - scene.proj_params.znear) * depth_map
        #colour_map = unfog(colour_map, full_depths, scene.fog_colour, scene.fog_density)

        frame_coords, frame_colours, tex = _depth_to_3D(depth_map, inv, inv_view, scene.proj_params.znear, scene.proj_params.zfar, colour_map)

        # First we must convert out of RGB and into screen space normals. Then,
        # transform screen space normals into world space.
        frame_normals = torch.from_numpy(frame.get_normal()).cuda().reshape([-1, 3])
        frame_normals = 2 * (frame_normals - 0.5)
        frame_normals[:,1] = -frame_normals[:,1]
        frame_normals[:,0] = -frame_normals[:,0]

        # Camera transform is rigid, so no need to inverse-transpose.
        normal_transform = inv_view[:3,:3]
        frame_normals = torch.matmul(frame_normals, normal_transform)
        frame_normals = frame_normals / torch.linalg.norm(frame_normals, dim=1).unsqueeze(-1)

        #frame_colours = tex

        coords = np.concatenate((coords, frame_coords.detach().to(torch.float64).cpu()), axis=0)
        colours = np.concatenate((colours, frame_colours.detach().to(torch.float64).cpu()), axis=0)
        normals = np.concatenate((normals, frame_normals.detach().to(torch.float64).cpu()), axis=0)
    return coords, colours, normals


def make_rotation(quaternions):
    ''' Make rotation matrix tensors from quaternion tensors.

        Return [batch_size, 3, 3] array of 3x3 rotation matrices. 
    '''
    assert(len(quaternions.shape) == 2)
    batch_size = quaternions.shape[0]

    # Ensure unit quaternion.
    quaternions = quaternions / torch.norm(quaternions, dim=1).unsqueeze(-1)

    # Convert quaternions to rotation matrices
    x, y, z, w = (quaternions[:, 0:1], quaternions[:, 1:2], 
            quaternions[:, 2:3], quaternions[:, 3:4])
    r00 = 1 - 2*(y**2 + z**2)
    r01 = 2*(x*y - w*z)
    r02 = 2*(x*z + w*y)
    r10 = 2*(x*y + w*z)
    r11 = 1 - 2*(x**2 + z**2)
    r12 = 2*(y*z - w*x)
    r20 = 2*(x*z - w*y)
    r21 = 2*(y*z + w*x)
    r22 = 1 - 2*(x**2 + y**2)
    rotation_matrices = torch.concat(
        (r00, r01, r02, r10, r11, r12, r20, r21, r22), dim=1
    )

    return rotation_matrices.reshape([batch_size, 3, 3])


def _infer_zparams(scene: RawSceneData):

    fl_x = scene.proj_params.fl_x
    fl_y = scene.proj_params.fl_y
    w = scene.proj_params.w
    h = scene.proj_params.h

    shape = scene.frames[0].get_depth().shape

    inv_proj = _inverse_projection_matrix(w, h, fl_x, fl_y, 
            torch.tensor(0.1, dtype=torch.float32), 
            torch.tensor(1000,  dtype=torch.float32)).T.cuda()

    # Camera translations and rotations grouped in a single tensor each.
    n_frames = len(scene.frames)
    original_camera_translations = torch.zeros(size=[n_frames, 3]).cuda()
    original_camera_rotations = torch.zeros(size=[n_frames, 3, 3]).cuda()
    for (cam_idx, frame) in enumerate(scene.frames):
        transform_matrix = torch.from_numpy(frame.transform).to(torch.float32).cuda()
        inv_view = transform_matrix
        original_camera_translations[cam_idx] = inv_view[3,:3]
        original_camera_rotations[cam_idx] = inv_view[:3,:3]

    # Create tensors for: 
    # - Number of frames that contain each feature.
    # - Recording the 2D pixel coords of each frame-feature.
    # - Recording the depth of each frame-feature.
    # - Recording the camera used by each frame-feature.
    feature_data = scene.feature_data
    n_features = scene.feature_data.n_features()
    n_frame_features = scene.feature_data.n_frame_features()
    feature_to_n_frames = torch.zeros(size=[n_features], dtype=torch.int64).cuda()
    frame_feature_pixel_coords = torch.zeros(size=[n_frame_features, 2], dtype=torch.float32).cuda()
    frame_feature_depths = torch.zeros(size=[n_frame_features], dtype=torch.float32).cuda()
    frame_feature_to_cam = torch.zeros(size=[n_frame_features], dtype=torch.int64).cuda()

    total_frame_feature_index = 0
    for feature_index in range(n_features):

        # Frame-features from the same feature are grouped together.
        matches_2D = feature_data.matches[feature_index]
        feature_to_n_frames[feature_index] = len(matches_2D) 

        # Iterate through frame-features of this point.
        for (frame_features, x, y) in matches_2D:
            y = 1440 - y

            frame = frame_features.frame
            depth = frame.get_depth()[int(round(y)), int(round(x)), 0]

            frame_feature_pixel_coords[total_frame_feature_index, 0] = x
            frame_feature_pixel_coords[total_frame_feature_index, 1] = y
            frame_feature_depths[total_frame_feature_index] = float(depth)
            frame_feature_to_cam[total_frame_feature_index] = frame.index
            total_frame_feature_index += 1

    # Create group tensor indicating where in frame_features, each feature 
    # starts and ends.
    feature_groups = torch.concat(
            (torch.tensor([0]).cuda(), torch.cumsum(feature_to_n_frames, dim=0)), 
            dim=0
    )

    # Near and far parameters.
    zstats_buf = torch.tensor([-0.8, 6.64]).cuda()
    zstats_param = torch.nn.Parameter(zstats_buf, requires_grad=True)

    # Camera position parameters.
    cam_pos_buf = torch.zeros(size=[len(scene.frames), 3]).cuda() 
    cam_param = torch.nn.Parameter(cam_pos_buf, requires_grad=True)

    # camera orientation parameters.
    cam_orientation_buf = torch.concat(
        (torch.zeros(size=[len(scene.frames), 3]), torch.ones(size=[len(scene.frames), 1])),
        axis=1
    ).cuda().detach()
    cam_orientation_param = torch.nn.Parameter(cam_orientation_buf, requires_grad=True)

    # Ground truth position and ground truth distance.
    fixed_cam_index = scene.lookup(ground_truth_frames[0]).index
    second_cam_index = scene.lookup(ground_truth_frames[1]).index

    # Orientation constraints.
    processed_orientation_constraints = []
    for constraint in orientation_constraints:
        processed = [scene.lookup(search_term) for search_term in constraint]
        processed_orientation_constraints.append(processed)

        first_frame = processed[0]
        for frame in processed[1:]:
            frame.transform[:3,:3] = first_frame.transform[:3,:3]
            original_camera_rotations[frame.index] = original_camera_rotations[first_frame.index]

    # Lineup constraints.
    processed_lineup_constraints = []
    for constraint in lineup_constraints:
        processed = [scene.lookup(search_term) for search_term in constraint]
        processed_lineup_constraints.append(processed)

    def objective_fn():
        optimizer.zero_grad()

        zstats = torch.exp(zstats_param)

        # Combine our adjusted parameters on top of the old camera parameters.
        total_translations = original_camera_translations + cam_param
        total_rotations = original_camera_rotations @ make_rotation(cam_orientation_param)
        total_transforms = torch.zeros(size=[n_frames, 4, 4]).cuda()
        total_transforms[:, 3, 3] = 1
        total_transforms[:, :3, :3] = total_rotations
        total_transforms[:, 3, :3] = total_translations

        inv_view = total_transforms[frame_feature_to_cam]

        coords_3D, _, _ = _depth_to_3D(frame_feature_depths, inv_proj, inv_view, 
                    zstats[0], zstats[1], pixel_coords_2D=frame_feature_pixel_coords, 
                    uses_depth_map=False, shape=shape)
        
        # Compute means of 3D points for each feature.
        coord_cumsum = torch.concat(
                (torch.tensor([[0, 0, 0]]).cuda(), torch.cumsum(coords_3D.to(torch.float64), dim=0)), 
                dim=0
        )
        means_3D = (coord_cumsum[feature_groups[1:]] - coord_cumsum[feature_groups[:-1]]).to(torch.float32) / feature_to_n_frames.view(-1, 1)

        # Compute deviation from means.
        repeats = torch.arange(0, n_features).cuda().repeat_interleave(feature_to_n_frames)
        extended_means = means_3D[repeats]
        deviations = torch.norm(coords_3D - extended_means, dim=1)

        # TODO: Weighting?
        loss = torch.mean(deviations)

        #err =  scene.feature_data.errs[point_index]
        #weight = 1 / (err + 1) ** 2

        loss.backward()
        #print(loss.detach())
        return loss
    
    optimizer = torch.optim.Adam([zstats_param, cam_param, cam_orientation_param], lr=2e-1)

    cam_param.requires_grad = False
    cam_orientation_param.requires_grad = False

    for t in range(0, 100):

        if t == 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 2e-2

        loss = optimizer.step(closure=objective_fn)
        print(f"Step {t}: ", loss)

    cam_param.requires_grad = True
    cam_orientation_param.requires_grad = True
    for t in range(0, 0 * 6000):

        if t == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 2e-1
        if t == 2000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 2e-2
        if t == 4000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 2e-3

        loss = optimizer.step(closure=objective_fn)
        if t % 10 == 0:
            print(f"Step {t}: ", loss)

        # Preserve some things about the first two cameras to prevent degenerate
        # solution:
        # - first cam's original position is taken as ground truth.
        # - second cam's distance to first cam is taken as ground truth.
        cam_param.data[fixed_cam_index] = 0
        original_dist = torch.norm(original_camera_translations[fixed_cam_index] - original_camera_translations[second_cam_index])
        second_to_first = ((original_camera_translations[fixed_cam_index] + cam_param.data[fixed_cam_index]) - 
                (original_camera_translations[second_cam_index] + cam_param.data[second_cam_index]))
        new_dist = torch.norm(second_to_first)
        cam_param.data[second_cam_index] = cam_param.data[second_cam_index] + second_to_first / new_dist * (new_dist - original_dist)
        second_to_first = ((original_camera_translations[fixed_cam_index] + cam_param.data[fixed_cam_index]) - 
                (original_camera_translations[second_cam_index] + cam_param.data[second_cam_index]))
        new_dist = torch.norm(second_to_first)
        
        # Enforce orientation constaints.
        # Sets of frames are known to share the exact same orientation as the
        # first frame in the constraint.
        for constraint in processed_orientation_constraints:
            first_orientation_param  = cam_orientation_param.data[constraint[0].index]
            for frame in constraint[1:]:
                cam_orientation_param.data[frame.index] = first_orientation_param

        # Enforce lineup constraints.
        # Sets of frames are known to be located exactly in the forward or 
        # backward direction from the first frame in the constraint.
        for constraint in processed_lineup_constraints:
            first_frame = constraint[0]
            start = original_camera_translations[first_frame.index] + cam_param.data[first_frame.index]
            rotation = original_camera_rotations[first_frame.index] @ make_rotation(cam_orientation_param[first_frame.index].unsqueeze(0))[0]
            dir = rotation[2,:]
            for frame in constraint[1:]:
                old_pos = original_camera_translations[frame.index] + cam_param.data[frame.index]
                z = (old_pos - start).dot(dir)
                new_pos = start + z * dir
                cam_param.data[frame.index] = new_pos - original_camera_translations[frame.index]
                pass

        if loss < 0.05:
            break


    zstats = torch.exp(zstats_param)
    scene.proj_params.znear = float(zstats[0].detach())
    scene.proj_params.zfar = float(zstats[1].detach())
    print(f"Selected znear and zfar: {scene.proj_params.znear} and {scene.proj_params.zfar}")
    
    # Permanently apply our adjusted camera positions.
    for (cam_idx, frame) in enumerate(scene.frames):
        frame.transform[:3,:3] = np.matmul(frame.transform[:3:,:3], make_rotation(cam_orientation_param[cam_idx].unsqueeze(0)).detach().cpu().numpy()) 
        frame.transform[3:,:3] = frame.transform[3:,:3] + cam_param[cam_idx].detach().cpu().numpy()

    pass
    

def _depth_to_3D(depth_data, inv_projection, inv_view, znear, zfar, 
        colour_map=None, pixel_coords_2D=None, uses_depth_map=True,
        shape=None):
    
    if uses_depth_map and shape is None:
        shape = depth_data.shape
    elif not uses_depth_map and shape is None:
        raise Exception("You must provide shape if not providing a depth map.")

    # You can optionally pass in pixel coordinates or they can be automatically
    # derived from the depth_map.
    if pixel_coords_2D is None:
        assert(uses_depth_map)
        y_coords, x_coords = torch.meshgrid(torch.arange(shape[0]).cuda(), torch.arange(shape[1]).cuda(), indexing='ij')
        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()
        pixel_coords_2D = torch.stack([x_coords, y_coords], axis=-1)
        B = shape[0] * shape[1]
    else:
        B = len(pixel_coords_2D)

    # Create homogenous normalized coordinates to match each depth map pixel.
    # Shape them into [B, 4].
    x_coords = 2 * pixel_coords_2D[:,0] / shape[1] - 1 + 1 / shape[1]
    y_coords = 2 * pixel_coords_2D[:,1] / shape[0] - 1 + 1 / shape[0]
    z_coords = torch.zeros_like(x_coords)
    projected_coords = torch.stack([x_coords, y_coords, z_coords, torch.ones_like(x_coords, dtype=torch.float32)], axis=-1)
    if uses_depth_map:
        depths = depth_data[pixel_coords_2D[:,1], pixel_coords_2D[:,0]][:,0]
    else:
        depths = depth_data

    # Calculate the view space coordinates
    view_coordinates = projected_coords @ inv_projection
    view_coordinates = view_coordinates / view_coordinates[:,3].reshape([B, 1])
    view_coordinates = view_coordinates / torch.linalg.norm(view_coordinates[:,:3], dim=1).unsqueeze(-1)
    z_want = znear + depths * (zfar - znear)
    z_want = -z_want
    view_coordinates = view_coordinates * (z_want / view_coordinates[:,2]).unsqueeze(-1)
    view_coordinates[:,3] = 1

    # Calculate the world space coordinates
    if len(inv_view.shape) == 2:
        point_homogeneous_world = view_coordinates @ inv_view
    else:
        point_homogeneous_world = view_coordinates.reshape([-1, 1, 4]) @ inv_view
        point_homogeneous_world = point_homogeneous_world.reshape([-1, 4])

    # Normalize to get the 3D coordinates
    point_3D = point_homogeneous_world[:,:3] / point_homogeneous_world[:,3].reshape([B, 1])

    if colour_map is not None:
        colours = colour_map.reshape([shape[0] * shape[1], 3])
    else:
        colours = depths

    #tex = torch.stack((x_coords * 0.5 + 0.5, y_coords * 0.5 + 0.5, flat_depth_map), dim=1)

    return point_3D, colours, None


def _perspective_projection_matrix(width, height, camera_angle_x, camera_angle_y, znear, zfar):

    aspect_ratio = width / height
    a = 1 / (aspect_ratio * torch.tan(torch.tensor(0.5 * camera_angle_y)))
    b = 1 / (torch.tan(torch.tensor(0.5 * camera_angle_y)))

    projection_matrix = torch.tensor([
        a, 0, 0, 0,
        0, b, 0,
        0, 0, -(zfar + znear) / (zfar - znear), -(2 * zfar * znear) / (zfar - znear),
        0, 0, -1, 0
    ]).reshape((4, 4))

    return projection_matrix


def _inverse_projection_matrix(width, height, camera_angle_x, camera_angle_y, znear, zfar):
    aspect_ratio = width / height
    a = 1 / (aspect_ratio * torch.tan(torch.tensor(0.5 * camera_angle_y)))
    b = 1 / (torch.tan(torch.tensor(0.5 * camera_angle_y)))
    c = -(zfar + znear) / (2 * zfar * znear)
    d = (zfar - znear) / (2 * zfar * znear)

    a = a.to(torch.float32)
    b = b.to(torch.float32)

    # Create the inverse projection matrix
    _n1 = torch.tensor(-1)
    _0 = torch.tensor(0)
    inv_projection_matrix = torch.stack([
        1 / a,   _0,     _0,  _0,
        _0,      1 / b,  _0,  _0,
        _0,      _0,     _0,  _n1,
        _0,      _0,     d,   c
    ]).reshape((4, 4))

    return inv_projection_matrix