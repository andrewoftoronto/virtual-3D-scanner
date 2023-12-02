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

# List of groups where each frame in the group includes a pixel that should be
# the same colour up close. They don't necessarily need to be at the same
# physical spatial location.
# Each such frame entry is (name, Y pixel, X pixel). Name can be a substring 
# of the colour map file name.
fog_inference_groups = [
    #[
    #    ('085', 1440 - 282, 402),
    #    ('058', 1440 - 523, 1348)
    #],
    [
        ('000', 720, 1280),
        ('004', 720, 1280),
        ('010', 720, 1280),
        ('011', 720, 1280)
    ]
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

    # Infer fog parameters from images that point to the same place.
    _infer_fog_params(scene, fog_inference_groups)

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

        # TODO: Compensate for fog.
        full_depths = scene.proj_params.znear + (scene.proj_params.zfar - scene.proj_params.znear) * depth_map
        colour_map = unfog(colour_map, full_depths, scene.fog_colour, scene.fog_density)

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

    zstats_buf = torch.tensor([-0.8, 6.64]).cuda()
    zstats_param = torch.nn.Parameter(zstats_buf, requires_grad=True)

    cam_pos_buf = torch.zeros(size=[len(scene.frames), 3]).cuda() 
    cam_param = torch.nn.Parameter(cam_pos_buf, requires_grad=True)

    cam_orientation_buf = torch.concat(
        (torch.zeros(size=[len(scene.frames), 3]), torch.ones(size=[len(scene.frames), 1])),
        axis=1
    ).cuda().detach()
    cam_orientation_param = torch.nn.Parameter(cam_orientation_buf, requires_grad=True)

    fl_x = scene.proj_params.fl_x
    fl_y = scene.proj_params.fl_y
    w = scene.proj_params.w
    h = scene.proj_params.h

    depth_map_cache = {}

    def objective_fn():
        optimizer.zero_grad()
        zstats = torch.exp(zstats_param)
        inv_proj = _inverse_projection_matrix(w, h, fl_x, fl_y, torch.tensor(0.1, dtype=torch.float32), torch.tensor(1000,  dtype=torch.float32)).T.cuda()

        id_to_point_predictions = {}
        for (cam_idx, (_, frame_features)) in enumerate(scene.feature_data.id_to_frame_features.items()):
            frame = frame_features.frame

            # Transform is view-to-world.
            transform_matrix = torch.from_numpy(frame.transform).to(torch.float32).cuda()
            inv_view = transform_matrix

            old_rotation = inv_view[:3,:3].clone()
            add_rotation = make_rotation(cam_orientation_param[cam_idx].unsqueeze(0))[0]
            new_rotation = old_rotation @ add_rotation
            inv_view[:3,:3] = new_rotation
            inv_view[3:,:3] = inv_view[3:,:3].clone() + cam_param[cam_idx]

            if cam_idx in depth_map_cache:
                depth_map = depth_map_cache[cam_idx]
            else:
                depth_map = torch.from_numpy(frame.get_depth()).cuda()
                depth_map_cache[cam_idx] = depth_map

            shape = depth_map.shape
            coords, _, tex = _depth_to_3D(depth_map, inv_proj, inv_view, zstats[0], zstats[1])    

            # Find depths of features.
            for (i, (x, y)) in enumerate(frame_features.coords_2D):
                id_3D = frame_features.point_3D_ids[i]
                if id_3D == -1:
                    continue

                point_index = scene.feature_data.id_to_point_index[id_3D]
                err =  scene.feature_data.errs[point_index]
                #if err >= 0.6:
                #    continue

                nearest_x = np.round(x).astype(np.int64)
                nearest_y = 1440 - np.round(y).astype(np.int64)
                point_3D = coords[nearest_y * shape[1] + nearest_x]
                if id_3D in id_to_point_predictions:
                    id_to_point_predictions[id_3D].append(point_3D)
                else:
                    id_to_point_predictions[id_3D] = [point_3D]

        # For each 3D point, compute standard deviation.
        loss = 0
        total_weight = 0
        for (id_3D, predictions) in id_to_point_predictions.items():
            point_index = scene.feature_data.id_to_point_index[id_3D]
            err =  scene.feature_data.errs[point_index]
            weight = 1 / (err + 1) ** 2

            predictions = torch.stack(predictions)
            mean = predictions.mean(axis=0).unsqueeze(0)
            loss = loss + torch.mean(torch.linalg.norm(predictions - mean, axis=1)) * weight
            total_weight += weight

        loss = loss / total_weight
        loss.backward()
        print(loss.detach())
        return loss
    
    optimizer = torch.optim.Adam([zstats_param, cam_param, cam_orientation_param], lr=2e-2)

    cam_param.requires_grad = False
    cam_orientation_param.requires_grad = False
    for t in range(0, 50):
        loss = optimizer.step(closure=objective_fn)

    for param_group in optimizer.param_groups:
        param_group['lr'] = 2e-3

    cam_param.requires_grad = True
    cam_orientation_param.requires_grad = True
    for t in range(0, 1000):
        loss = optimizer.step(closure=objective_fn)

    zstats = torch.exp(zstats_param)
    scene.proj_params.znear = float(zstats[0].detach())
    scene.proj_params.zfar = float(zstats[1].detach())
    print(f"Selected znear and zfar: {scene.proj_params.znear} and {scene.proj_params.zfar}")
    
    # Permanently apply our adjusted camera positions.
    for (cam_idx, (_, frame_features)) in enumerate(scene.feature_data.id_to_frame_features.items()):
        frame = frame_features.frame
        frame.transform[:3,:3] = np.matmul(frame.transform[:3:,:3], make_rotation(cam_orientation_param[cam_idx].unsqueeze(0)).detach().cpu().numpy()) 
        frame.transform[3:,:3] = frame.transform[3:,:3] + cam_param[cam_idx].detach().cpu().numpy()

    pass


def unfog(foggy_colour, depth, fog_colour, fog_density):
    #fog_density = torch.exp(fog_density_log)
    if len(foggy_colour.shape) == 3:
        fog_colour = fog_colour.reshape([1, 1, 3])
        fog_density = fog_density.reshape([1, 1, 1])
    recovered_colour = foggy_colour - (
        fog_colour * (1 - torch.exp(-depth * fog_density)) / 
        torch.exp(-depth * fog_density)
    )
    return recovered_colour

def fog(initial_colour, depth, fog_colour, fog_density_log):
    fog_density = torch.exp(fog_density_log)
    interpolant = torch.exp(-depth * fog_density)
    predicted_colour = initial_colour * interpolant + fog_colour * (1 - interpolant)
    return predicted_colour

def _infer_fog_params(scene: RawSceneData, image_groups: List[List[str]]):

    # Assemble frames that show the same colour.
    frame_groups = []
    for image_group in image_groups:
        frame_group = []
        for (image_name, y, x) in image_group:
            
            # TODO: This looks stupid and can use hashmaps instead.
            selected_frame = None
            for frame in chain(scene.frames, scene.extra_frames):
                if image_name in frame.colour_path:
                    selected_frame = frame
                    break
            if selected_frame is None:
                raise Exception(f"No match found for fog inference image: {image_name}")
            frame_group.append([selected_frame, y, x])

        frame_groups.append(frame_group)

    fog_colour_buf = torch.tensor([0.5, 0.5, 0.5]).cuda()
    fog_colour_param = torch.nn.Parameter(fog_colour_buf, requires_grad=True)

    fog_density_log_buf = torch.tensor([-5.0]).cuda()
    fog_density_log_param = torch.nn.Parameter(fog_density_log_buf, requires_grad=True)

    initial_colours_buf = torch.ones(size=[len(frame_groups), 3]).cuda() * 0.5
    initial_colours_param = torch.nn.Parameter(initial_colours_buf, requires_grad=True)

    def objective_fn():
        optimizer.zero_grad()

        loss = 0
        total_predictions = 0
        for (i, frame_group) in enumerate(frame_groups):

            for (frame, y, x) in frame_group:

                # TODO: Codify this better.
                colour_map = torch.from_numpy(frame.get_colour()).cuda()
                colour_map = torch.flip(colour_map, dims=[0]) / 255.0

                depth_map = torch.from_numpy(frame.get_depth()).cuda()

                expected_colour = colour_map[y][x][:3]
                true_depth = scene.proj_params.znear + depth_map[y][x][0] * (scene.proj_params.zfar - scene.proj_params.znear)
                predicted_colour = fog(initial_colours_param[i], true_depth, fog_colour_param, fog_density_log_param)

                loss += torch.nn.MSELoss()(predicted_colour, expected_colour)
                total_predictions += 1

        loss = loss / total_predictions
        loss.backward()
        print(loss.detach())

        return loss
    
    optimizer = torch.optim.Adam([fog_colour_param, fog_density_log_param, initial_colours_param], lr=1e-1)

    for t in range(0, 500):
        loss = optimizer.step(closure=objective_fn)

        # Project initial colour to be non-negative.
        initial_colours_param.data = torch.maximum(initial_colours_param.data, torch.tensor(0).cuda())

        if loss < 0.00015:
            break

    scene.fog_colour = fog_colour_param.data
    scene.fog_density = torch.exp(fog_density_log_param.data)
    print(f"Selected fog colour and density: {scene.fog_colour} and {scene.fog_density}")
    pass
    

def _depth_to_3D(depth_map, inv_projection, inv_view, znear, zfar, colour_map=None):
    shape = depth_map.shape
    B = shape[0] * shape[1]

    flat_depth_map = depth_map.reshape([B, 3])[:,0]

    # Create homogenous normalized coordinates to match each depth map pixel.
    # Shape them into [B, 4].
    shape = depth_map.shape
    y_coords, x_coords = torch.meshgrid(torch.arange(shape[0]).cuda(), torch.arange(shape[1]).cuda(), indexing='ij')
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    x_coords = 2 * x_coords / shape[1] - 1 + 1 / shape[1]
    y_coords = 2 * y_coords / shape[0] - 1 + 1 / shape[0]
    #x_coords = -x_coords
    z_coords = torch.zeros_like(x_coords)
    pixel_coordinates = torch.stack([x_coords, y_coords, z_coords, torch.ones_like(x_coords, dtype=torch.float32)], axis=-1)

    # Calculate the view space coordinates
    view_coordinates = pixel_coordinates @ inv_projection
    view_coordinates = view_coordinates / view_coordinates[:,3].reshape([B, 1])
    view_coordinates = view_coordinates / torch.linalg.norm(view_coordinates[:,:3], dim=1).unsqueeze(-1)
    z_want = znear + flat_depth_map * (zfar - znear)
    z_want = -z_want
    view_coordinates = view_coordinates * (z_want / view_coordinates[:,2]).unsqueeze(-1)
    view_coordinates[:,3] = 1

    # Calculate the world space coordinates
    point_homogeneous_world = view_coordinates @ inv_view

    # Normalize to get the 3D coordinates
    point_3D = point_homogeneous_world[:,:3] / point_homogeneous_world[:,3].reshape([B, 1])

    if colour_map is not None:
        colours = colour_map.reshape([shape[0] * shape[1], 3])
    else:
        colours = flat_depth_map

    tex = torch.stack((x_coords * 0.5 + 0.5, y_coords * 0.5 + 0.5, flat_depth_map), dim=1)

    return point_3D, colours, tex


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