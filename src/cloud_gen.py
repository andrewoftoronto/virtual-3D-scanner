from typing import List, Tuple
import torch
import numpy as np
from .source_loader import RawSceneData

# List of groups where each frame in the group includes a pixel that should 
# occupy the same 3D spatial location.
# Each such frame entry is (name, Y pixel, X pixel). Name can be a substring 
# of the colour map file name.
infer_groups = [
    [
        ('054', 533, 2163),
        ('058', 1007, 1135)
    ]
]


def cloud_gen(scene: RawSceneData):
    ''' Generate a point cloud from the given raw scene data. '''

    znear = 0.0001
    zfar = 10000

    fl_x = scene.proj_params.fl_x
    fl_y = scene.proj_params.fl_y
    cx = scene.proj_params.cx
    cy = scene.proj_params.cy
    w = scene.proj_params.w
    h = scene.proj_params.h

    # Infer zparams using two images that point to the same place.
    _infer_zparams(scene, infer_groups)

    inv = _inverse_projection_matrix(w, h, fl_x, fl_y, torch.tensor(scene.proj_params.znear), torch.tensor(scene.proj_params.zfar)).cuda().T

    coords = np.zeros([0,3], dtype=np.float64)
    colours = np.zeros([0,3], dtype=np.float64)
    normals = np.zeros([0,3], dtype=np.float64)
    for (i, frame) in enumerate(scene.frames):
        #onlys = ['54', '58']
        onlys = ['54', '58', '66', '69', '85']
        if not any([only in frame.colour_path for only in onlys]):
            continue
        #if i >= 20:
        #    continue

        transform_matrix = torch.from_numpy(frame.transform).to(torch.float32).cuda()
        inv_view = transform_matrix.T

        depth_map = torch.from_numpy(frame.get_depth()).cuda()
        colour_map = torch.from_numpy(frame.get_colour()).cuda()
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

        frame_colours = tex

        coords = np.concatenate((coords, frame_coords.detach().to(torch.float64).cpu()), axis=0)
        colours = np.concatenate((colours, frame_colours.detach().to(torch.float64).cpu()), axis=0)
        normals = np.concatenate((normals, frame_normals.detach().to(torch.float64).cpu()), axis=0)
    return coords, colours, normals


def _infer_zparams(scene: RawSceneData, image_groups: List[List[str]]):
     
    # Assemble frames that point to the same thing.
    frame_groups = []
    for image_group in image_groups:
        frame_group = []
        for (image_name, y, x) in image_group:
            
            # TODO: This looks stupid and can use hashmaps instead.
            selected_frame = None
            for frame in scene.frames:
                if image_name in frame.colour_path:
                    selected_frame = frame
                    break
            if selected_frame is None:
                raise Exception(f"No match found for zparam inference image: {image_name}")
            frame_group.append([selected_frame, y, x])

        frame_groups.append(frame_group)

    zstats_buf = torch.tensor([-100, 8.6996]).cuda()
    zstats_param = torch.nn.Parameter(zstats_buf, requires_grad=True)

    fl_x = scene.proj_params.fl_x
    fl_y = scene.proj_params.fl_y
    w = scene.proj_params.w
    h = scene.proj_params.h

    def objective_fn():
        optimizer.zero_grad()
        zstats = torch.exp(zstats_param)
        inv_proj = _inverse_projection_matrix(w, h, fl_x, fl_y, torch.tensor(0.1), torch.tensor(1000)).T.cuda()
        loss = 0

        i = 0
        total_groups = 0
        for frame_group in frame_groups:

            results = []
            for (frame, y, x) in frame_group:
                transform_matrix = torch.from_numpy(frame.transform).to(torch.float32).cuda()

                # Looks like we were given world-to-view.
                inv_view = transform_matrix.T

                depth_map = frame.get_depth()
                shape = depth_map.shape
                #depth_map = depth_map[depth_map.shape[0]//2,depth_map.shape[1]//2]
                #depth_map = depth_map.reshape([1, 1, 3])
                depth_map = torch.from_numpy(depth_map).cuda()
                coords, _, tex = _depth_to_3D(depth_map, inv_proj, inv_view, zstats[0], zstats[1])

                #print(frame.colour_path, "Cam: ", inv_view[3,:], "Depth:", depth_map, "Forward:", inv_view[2,:])
                #if i == 0:
                #    print("Camera coord", inv_view[3,:], "World coord: ", coords, "view-coords", vc, "depth", depth_map)

                results.append(coords[y * shape[1] + x])
            i += 1

            # Compare each possible pair of result points.
            pair_losses = []
            for (i1, p1) in enumerate(results):
                for p2 in results[i1 + 1:]:
                    pair_loss = torch.sum((p1 - p2) ** 2)
                    pair_losses.append(pair_loss)
            total_groups += 1
            group_loss = torch.mean(torch.stack(pair_losses))
            loss = loss + group_loss
        loss = loss / total_groups
        loss.backward()
        print(loss.detach())
        return loss
    
    optimizer = torch.optim.Adam([zstats_param], lr=1e-3)
    zbest = [-0.5108256237659907, 6.50812574076698]
    zstats_buf[0] = zbest[0]
    zstats_buf[1] = zbest[1]

    for t in range(0, 200):
        loss = optimizer.step(closure=objective_fn)
        if loss < 0.0007:
            break

    zstats = torch.exp(zstats_param)
    scene.proj_params.znear = float(zstats[0].detach())
    scene.proj_params.zfar = float(zstats[1].detach())
    print(f"Selected znear and zfar: {scene.proj_params.znear} and {scene.proj_params.zfar}")
    #scene.proj_params.znear = 0.6
    #scene.proj_params.zfar = 770
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
        colours = colour_map.reshape([shape[0] * shape[1], 4])[:,:3]
    else:
        colours = flat_depth_map

    tex = torch.stack((x_coords * 0.5 + 0.5, y_coords * 0.5 + 0.5, flat_depth_map), dim=1)

    return point_3D, colours, tex


def _perspective_projection_matrix(width, height, camera_angle_x, camera_angle_y, znear, zfar):

    aspect_ratio = width / height
    top = torch.tan(torch.tensor(0.5 * camera_angle_y)) * znear
    bottom = -top
    right = top * aspect_ratio
    left = -right

    projection_matrix = torch.tensor([
        (2 * znear) / (right - left), 0, (right + left) / (right - left), 0,
        0, (2 * znear) / (top - bottom), (top + bottom) / (top - bottom), 0,
        0, 0, -(zfar + znear) / (zfar - znear), -(2 * zfar * znear) / (zfar - znear),
        0, 0, -1, 0
    ]).reshape((4, 4))

    return projection_matrix


def _inverse_projection_matrix(width, height, camera_angle_x, camera_angle_y, znear, zfar):
    aspect_ratio = width / height
    top = torch.tan(torch.tensor(0.5 * camera_angle_y)) * znear
    bottom = -top
    right = top * aspect_ratio
    left = -right

    a = (2 * znear) / (right - left)
    b = (2 * znear) / (top - bottom)
    c = -(zfar + znear) / (2 * zfar * znear)
    d = (zfar - znear) / (2 * zfar * znear)

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