''' Functions for projecting, transforming scene data, etc. '''
from typing import Optional
from .source_loader import RawSceneData
from .source_loader import Frame
import torch

def unproject(scene_data: RawSceneData, frame: Frame, compute_device="cuda",
            max_dist: Optional[float]=None):
    ''' Unproject the given frame into world space. '''

    proj = make_projection_matrix(scene_data)
    inv_proj = proj.inverse().to(compute_device)
    inv_view = torch.from_numpy(frame.transform).to(compute_device)

    depths = torch.from_numpy(frame.get_depth()).to(compute_device)
    shape = depths.shape
    world_positions = depth_to_3D(depths, inv_projection=inv_proj, 
            inv_view=inv_view, pixel_coords_2D=None, uses_depth_map=True, 
            shape=shape)
    del depths

    # Enforce maximum dist from origin.
    if max_dist is not None:
        dists = torch.linalg.norm(world_positions, dim=1, keepdim=True)
        maintain_mask = dists < max_dist
        world_positions = (
                maintain_mask * world_positions +
                ~maintain_mask * world_positions / dists * max_dist
        )

    return world_positions


def unproject_ray(scene_data: RawSceneData, frame: Frame, compute_device="cuda"):
    ''' Create tensor of all world-space rays emitted from each pixel of the 
        given frame.
      
        Unlike unproject, this does not take into account the actual depth of
        each frame. Instead it just creates a ray based on the frame's 
        transformation and the camera projection parameters.
    '''

    proj = make_projection_matrix(scene_data)
    inv_proj = proj.inverse().to(compute_device)
    inv_view = torch.from_numpy(frame.transform).to(compute_device)
    shape = frame.get_colour().shape

    depth_data = torch.zeros(size=(shape[0], shape[1]), dtype=torch.float32).to(compute_device)
    ray_starts = depth_to_3D(depth_data, inv_proj, inv_view)

    depth_data = 0.1 * torch.ones(size=(shape[0], shape[1]), dtype=torch.float32).to(compute_device)
    ray_ends = depth_to_3D(depth_data, inv_proj, inv_view)
    ray_dirs = ray_ends - ray_starts
    ray_dirs /= torch.linalg.norm(ray_dirs, dim=1, keepdim=True)
    return ray_starts, ray_dirs


def depth_to_3D(depth_data, inv_projection, inv_view,
        pixel_coords_2D=None, uses_depth_map=True, shape=None):
    ''' General and free-form function for computing 3D world space positions 
        from a set of pixel coordinates and depth values. '''
    
    if pixel_coords_2D is None:
        assert(uses_depth_map)

    if uses_depth_map and shape is None:
        shape = depth_data.shape
    elif not uses_depth_map and shape is None:
        raise Exception("You must provide shape if not providing a depth map.")

    # You can pass in pixel coordinates or they are automatically derived based
    # on shape.
    pixel_coords_2D, x_coords, y_coords = _make_proj_coords(
            inv_projection, shape, pixel_coords_2D)
    if uses_depth_map:
        depths = depth_data.reshape([shape[0] * shape[1]])
    else:
        depths = depth_data
    B = len(x_coords)

    # Create homogenous normalized coordinates to match each depth map pixel.
    # Shape them into [B, 4].
    z_coords = 2 * depths - 1
    projected_coords = torch.stack([x_coords, y_coords, z_coords, torch.ones_like(x_coords, dtype=torch.float32)], axis=-1)

    # Calculate the view space coordinates
    view_coordinates = projected_coords @ inv_projection
    view_coordinates = view_coordinates / view_coordinates[:,3].reshape([B, 1])

    # Calculate the world space coordinates
    if len(inv_view.shape) == 2:
        point_homogeneous_world = view_coordinates @ inv_view
    else:
        point_homogeneous_world = view_coordinates.reshape([-1, 1, 4]) @ inv_view
        point_homogeneous_world = point_homogeneous_world.reshape([-1, 4])

    # Normalize to get the 3D coordinates
    point_3D = point_homogeneous_world[:,:3] / point_homogeneous_world[:,3].reshape([B, 1])

    return point_3D


def _make_proj_coords(inv_proj, shape, pixel_coords_2D=None):
    ''' Make 2D projection coordinates based either based entirely on the given
        projection surface shape or a list of actual pixel coordinates. 
        
        inv_proj: inverse projection matrix.
        shape: 2D shape of the projection surface in pixels.
        pixel_coords_2D: optional list of pixel coordinates to use. Otherwise
            creates a list that includes every pixel based on shape.
        return: (
            2D pixel coordinates tensor (in pixels), 
            x_coords (x-axis projection coordinates), 
            y_coords (y-axis projection coordinates)
        )
        '''

    # You can optionally pass in pixel coordinates or they can be automatically
    # derived from the depth_map.
    if pixel_coords_2D is None:
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
    return pixel_coords_2D, x_coords, y_coords


def make_projection_matrix(scene_data: RawSceneData):
    ''' Creates a device agnostic projection matrix tensor for use in 
        multiplying like so:
        Vview = Vproj x InvProj '''

    width = scene_data.proj_params.w
    height = scene_data.proj_params.h
    camera_angle_x = scene_data.proj_params.fl_x
    camera_angle_y = scene_data.proj_params.fl_y
    znear = scene_data.proj_params.znear
    zfar = scene_data.proj_params.zfar

    aspect_ratio = width / height
    a = 1 / (torch.tan(torch.tensor(0.5 * camera_angle_x)))
    b = 1 / (torch.tan(torch.tensor(0.5 * camera_angle_y)))

    projection_matrix = torch.tensor([
        a, 0, 0, 0,
        0, b, 0, 0,
        0, 0, -(zfar + znear) / (zfar - znear), -(2 * zfar * znear) / (zfar - znear),
        0, 0, -1, 0
    ]).reshape((4, 4))

    return projection_matrix.T


def make_inv_projection_matrix(scene_data: RawSceneData):
    ''' Creates a device agnostic projection matrix tensor for use in 
        multiplying like so:
        Vview = Vproj x InvProj '''
    raise Exception("Warning broken")

    width = scene_data.proj_params.w
    height = scene_data.proj_params.h
    camera_angle_x = scene_data.proj_params.fl_x
    camera_angle_y = scene_data.proj_params.fl_y
    znear = scene_data.proj_params.znear
    zfar = scene_data.proj_params.zfar

    aspect_ratio = width / height
    a = 1 / (torch.tan(torch.tensor(0.5 * camera_angle_x)))
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

    return inv_projection_matrix.T

