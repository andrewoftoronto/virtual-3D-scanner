import torch
import numpy as np
from .source_loader import RawSceneData

def run(scene_data: RawSceneData):
    proj_params = scene_data.proj_params

    d_frame_sets = [
            ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007']
    ]

    LEARN_ZFAR = True
    znear_raw = 0.10837346456903406
    zfar_raw = 999999999

    def opt_iteration():
        znear = torch.tensor(znear_raw)
        zfar = torch.tensor(zfar_raw)
        proj = _perspective_projection_matrix(proj_params.w, proj_params.h, proj_params.fl_x, proj_params.fl_y, znear, zfar).cuda()
        inv_proj = torch.linalg.inv(proj)

        linear_depth_sets = []
        camera_pos_sets = []
        for (i, set) in enumerate(d_frame_sets):
            linear_depths = []
            camera_poses = []

            for frame_tag in set:
                frame = scene_data.lookup(frame_tag)
                depth = frame.get_depth()[720, 1280, 0]
                depth = depth * 2 - 1

                # Unproject.
                proj_coord = torch.tensor([[0, 0, depth, 1]]).cuda().to(torch.float32)
                view_coord = torch.matmul(proj_coord, inv_proj)
                view_coord = view_coord / view_coord[:,3]
                linear_depths.append(view_coord[:,2])
                camera_poses.append(frame.get_position())

            linear_depth_sets.append(linear_depths)
            camera_pos_sets.append(camera_poses)

        return linear_depth_sets, camera_pos_sets

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

        linear_depth_sets, camera_coord_sets = opt_iteration()

        score = 0
        for set_index in range(0, len(linear_depth_sets)):
            camera_coords = camera_coord_sets[set_index]
            linear_depths = linear_depth_sets[set_index]
            for i in range(0, len(linear_depths)):
                for j in range(i+1, len(linear_depths)):
                    d0 = linear_depths[i]
                    d1 = linear_depths[j]
                    c0 = camera_coords[i]
                    c1 = camera_coords[j]
                    vdist = torch.abs(d0 - d1).detach().cpu()
                    cdist = np.linalg.norm(c0 - c1)
                    err = np.abs(vdist - cdist)
                    score += err

        record = (score, znear_raw, zfar_raw)
        records.append(record)
        if best is None:
            best = record
        elif record[0] < best[0]:
            best = record
    
    scene_data.proj_params.znear = best[1]
    scene_data.proj_params.zfar = best[2]
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