import torch
import numpy as np
from .source_loader import RawSceneData

def run(scene_data: RawSceneData):
    optimize_zstats(scene_data)

    # Plot z vs depth
    d_frames = []
    for frame_name in d_frames:
        pass
        #z = 

    # Estimate z-near and z-far.

    pass

def optimize_zstats(scene_data: RawSceneData):
    proj_params = scene_data.proj_params

    #d_frame_sets = [
    #        ['00', '01', '02', '03', '04', '05']
    #]
    d_frame_sets = [
            ['0000', '0001', '0002', '0003']
    ]

    LEARN_ZFAR = False
    znear_raw = 0.08526150593834063
    zfar_raw = 9999999
    world_coords = np.array([
        [-3.10613866,  -2.64558899,   2.86173452],
        [-24.46329244,   0.91266925,   2.55176874]
    ])

    '''start = None
    for frame_tag in d_frames:
        frame = scene_data.lookup(frame_tag)
        pos = frame.get_position()
        forward = frame.get_forward()
        depth = frame.get_depth()[720, 1280, 0]
        z = pos.dot(forward)

        if start is None:
            start = z
        print(z - start, depth)'''

    def opt_iteration():
        znear = torch.tensor(znear_raw)
        zfar = torch.tensor(zfar_raw)
        proj = _perspective_projection_matrix(proj_params.w, proj_params.h, proj_params.fl_x, proj_params.fl_y, znear, zfar).cuda()
        inv_proj = torch.linalg.inv(proj)

        linear_depth_sets = []
        view_coord_sets = []
        camera_pos_sets = []
        for (i, set) in enumerate(d_frame_sets):
            linear_depths = []
            view_coords = []
            camera_poses = []

            for frame_tag in set:
                frame = scene_data.lookup(frame_tag)
                depth = frame.get_depth()[720, 1280, 0]
                depth = depth * 2 - 1
                #depth = depth ** 2.2

                # Linearize.
                #linear_depth = 2 * zfar * znear / (zfar + znear - depth * (zfar - znear))

                # Unproject.
                proj_coord = torch.tensor([[0, 0, depth, 1]]).cuda().to(torch.float32)
                view_coord = torch.matmul(proj_coord, inv_proj)
                view_coord = view_coord / view_coord[:,3]
                linear_depths.append(view_coord[:,2])

                #linear_depths.append(linear_depth)
                view_coords.append(view_coord[:,:3])
                camera_poses.append(frame.get_position())

            linear_depth_sets.append(linear_depths)
            view_coord_sets.append(view_coords)
            camera_pos_sets.append(camera_poses)

        return linear_depth_sets, view_coord_sets, camera_pos_sets

    best = None
    records = []
    for i in range(0, 10000):

        if best is not None:
            if np.random.uniform() < 0.5:
                znear_raw = np.random.uniform(1e-8, 2)
                if LEARN_ZFAR:
                    zfar_raw = np.random.uniform(10, 1000)
            else:
                if np.random.uniform() < 0.5:
                    if np.random.uniform() < 0.5:
                        znear_raw = best[1] + best[1] * np.random.uniform(-1e-4, 1e-4)
                        pass
                    else:
                        znear_raw = best[1] + best[1] * np.random.uniform(-1e-2, 1e-2)
                        pass
                else:
                    znear_raw = np.random.uniform(1e-8, 2)
                    pass
                if LEARN_ZFAR:
                    if np.random.uniform() < 0.5:
                        zfar_raw = best[2] + np.random.uniform(-10, 10)
                    else:
                        if np.random.uniform() < 0.5:
                            zfar_raw = np.random.uniform(10, 1e9)
                        else:
                            zfar_raw = np.random.uniform(10, 1e9)
            best_coords = best[3].copy()
            for j in range(0, len(world_coords)):
                for c in range(0, 3):
                    if np.random.uniform() < 0.25:
                        if np.random.uniform() < 0.5:
                            world_coords[j][c] = best_coords[j][c] + 0.01 * np.random.uniform()
                        else:
                            world_coords[j][c] = best_coords[j][c] + np.random.uniform()

        linear_depth_sets, view_coord_sets, camera_coord_sets = opt_iteration()

        world_coord_score = 0
        '''for world_coords in world_coord_sets:
            world_coords = np.array(world_coords)
            coord_mean = np.mean(world_coords, axis=1)
            dists = np.linalg.norm(world_coords - coord_mean, axis=1)
            mean_dist = np.mean(dists)
            world_coord_score += mean_dist / 2'''

        score = 0
        for set_index in range(0, len(view_coord_sets)):
            view_coords = view_coord_sets[set_index]
            camera_coords = camera_coord_sets[set_index]
            linear_depths = linear_depth_sets[set_index]
            for i in range(0, len(view_coords)):
                for j in range(i+1, len(view_coords)):
                    d0 = linear_depths[i]
                    d1 = linear_depths[j]
                    v0 = view_coords[i]
                    v1 = view_coords[j]
                    c0 = camera_coords[i]
                    c1 = camera_coords[j]
                    vdist = torch.abs(d0 - d1).detach().cpu()
                    cdist = np.linalg.norm(c0 - c1)
                    err = np.abs(vdist - cdist)
                    score += err
                    #print(torch.maximum(d0, d1), cdist, vdist, err)

        record = (score, znear_raw, zfar_raw, world_coords.copy())
        records.append(record)
        if best is None:
            best = record
        elif record[0] < best[0]:
            best = record
    records.sort(key=lambda x: x[0])
    print(records[0:10])

    pass
    

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

def _perspective_projection_matrix(width, height, camera_angle_x, camera_angle_y, znear, zfar):

    aspect_ratio = width / height
    a = 1 / (aspect_ratio * torch.tan(torch.tensor(0.5 * camera_angle_y)))
    b = 1 / (torch.tan(torch.tensor(0.5 * camera_angle_y)))

    projection_matrix = torch.tensor([
        a, 0, 0, 0,
        0, b, 0, 0,
        0, 0, -(zfar + znear) / (zfar - znear), -(2 * zfar * znear) / (zfar - znear),
        0, 0, -1, 0
    ]).reshape((4, 4))

    return projection_matrix.T

def _perspective_projection_matrix_d3D(width, height, camera_angle_x, camera_angle_y, znear, zfar):

    aspect_ratio = width / height
    a = 1/torch.tan(torch.tensor(camera_angle_x*0.5))
    b = 1/torch.tan(torch.tensor(camera_angle_y*0.5))
    Q = zfar/(zfar - znear)

    projection_matrix = torch.tensor([
        a, 0, 0, 0,
        0, b, 0, 0,
        0, 0, Q, 1,
        0, 0, -Q * znear, 0
    ]).reshape((4, 4))

    return projection_matrix