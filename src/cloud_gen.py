from typing import List, Tuple
from itertools import chain
import torch
import numpy as np
from .source_loader import RawSceneData
import open3d as o3d


class ColinearConstraint:
    def __init__(self, parent_keyword: str, 
                 direction, 
                 children: List[str|Tuple[str, bool]]):
        self.parent = parent_keyword
        self.direction = np.array(direction, dtype=np.float32)
        self.children = children

    def resolve_names(self, scene_data: RawSceneData):
        ''' Resolve name keywords into actual frame objects. '''

        self.parent = scene_data.lookup(self.parent)

        # Resolve children. Each may be a tuple or just name.
        for (i, child) in enumerate(self.children):
            if isinstance(child, tuple):
                keyword, learnable_dist = child[0], child[1]
            else:
                keyword, learnable_dist = child, True
            frame = scene_data.lookup(keyword)
            self.children[i] = (frame, learnable_dist)


class CameraInferenceContext:

    def __init__(self, 
            scene_data: RawSceneData, 
            fixed_pos_constraints,
            orientation_constraints, 
            colinear_constraints: List[ColinearConstraint]):
        self.scene_data: RawSceneData = scene_data
        self.fixed_pos_constraints = fixed_pos_constraints
        self.orientation_constraints = orientation_constraints
        self.colinear_constraints = colinear_constraints

        # By default, assume all cameras are unconstrained and that their
        # translation vector is a learnable parameter.
        self.learnable_vector_bitmap = np.ones(shape=len(scene_data.frames))

        self.cam_to_orientation_source = None
        self.orientation_params = None

    def get_orientation(self, camera_index):
        source_index = self.cam_to_orientation_source[camera_index]
        params = self.orientation_params[source_index] 
        return make_rotation(params.unsqueeze(0))[0]

def cloud_gen(scene: RawSceneData):
    ''' Generate a point cloud from the given raw scene data. '''

    fl_x = scene.proj_params.fl_x
    fl_y = scene.proj_params.fl_y
    cx = scene.proj_params.cx
    cy = scene.proj_params.cy
    w = scene.proj_params.w
    h = scene.proj_params.h

    proj = _perspective_projection_matrix(w, h, fl_x, fl_y, torch.tensor(scene.proj_params.znear, dtype=torch.float32), torch.tensor(scene.proj_params.zfar, dtype=torch.float32)).cuda().T
    inv = torch.linalg.inv(proj)

    coords = np.zeros([0,3], dtype=np.float64)
    colours = np.zeros([0,3], dtype=np.float64)
    normals = np.zeros([0,3], dtype=np.float64)
    clouds = []
    for (i, frame) in enumerate(scene.frames):
        if np.random.uniform() > 0.1:
            continue

        transform_matrix = torch.from_numpy(frame.transform).to(torch.float32).cuda()
        inv_view = transform_matrix

        depth_map = torch.from_numpy(frame.get_depth()).cuda()
        #depth_map = torch.flip(depth_map, dims=[0])

        # TODO: Codify this better.
        colour_map = torch.from_numpy(frame.get_colour()).cuda()
        colour_map = colour_map[:,:,:3] / 255

        frame_coords, frame_colours, tex = _depth_to_3D(depth_map, inv, inv_view, scene.proj_params.znear, scene.proj_params.zfar, colour_map)
        n_points = len(frame_coords)

        shape = frame.get_depth().shape
        c = frame_coords.reshape([shape[0], shape[1], 3])
        t1 = c[1:,:-1] - c[:-1,:-1]
        t2 = c[:-1,1:] - c[:-1,:-1]
        n = torch.cross(t2, t1, dim=2)
        n = n / torch.norm(n, dim=2).unsqueeze(-1)

        frame_coords = c[:-1,:-1]
        frame_normals = n[:-1,:-1]
        frame_colours = frame_colours.reshape([shape[0], shape[1], 3])[:-1,:-1]

        frame_coords = frame_coords.reshape([-1, 3]).detach().cpu().numpy()
        frame_normals = frame_normals.reshape([-1, 3]).detach().cpu().numpy()
        frame_colours = frame_colours.reshape([-1, 3]).detach().cpu().numpy()

        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(frame_coords.detach().cpu().numpy())
        #pcd.normals = o3d.utility.Vector3dVector(frame_normals.detach().cpu().numpy())
        #pcd.colors = o3d.utility.Vector3dVector(frame_colours.detach().cpu().numpy())
        '''if prev_cloud is not None:
            icp_result = o3d.pipelines.registration.registration_icp(
                pcd, prev_cloud,
                max_correspondence_distance=0.05,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
            pcd.transform(icp_result.transformation)
        prev_cloud = pcd'''

        onlys = ['000', '001', '009', '017', '002', '004', '023']
        excludes = ['000']
        #if any([only in frame.colour_path for only in onlys]):
        #if not any([exclude in frame.colour_path for exclude in excludes]):
        if True:
            #capture_clouds.append(pcd)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(frame_coords)
            pcd.normals = o3d.utility.Vector3dVector(frame_normals)
            pcd.colors = o3d.utility.Vector3dVector(frame_colours)
            clouds.append(pcd)
        
    #o3d.visualization.draw_geometries(capture_clouds)
    #exit(0)

    return clouds, coords, colours, normals


def apply_fixed_pos_constraints(context: CameraInferenceContext):
    for frame_keyword in context.fixed_pos_constraints:
        frame = context.scene_data.lookup(frame_keyword)
        context.learnable_vector_bitmap[frame.index] = 0


def apply_orientation_constraints(context: CameraInferenceContext):
    n_frames = len(context.scene_data.frames)

    # Identifies frames participating in a constraint so we can detect invalid
    # data.
    constrained_orientation_bitmap = np.zeros(shape=[n_frames], dtype=bool)

    for orientation_constraint in context.orientation_constraints:
        for (i, member_keyword) in enumerate(orientation_constraint):
            frame = context.scene_data.lookup(member_keyword)
            orientation_constraint[i] = frame

            if constrained_orientation_bitmap[frame.index] != 0:
                raise Exception(f"Frame {frame.index} participates in multiple orientation constraints.")
            constrained_orientation_bitmap[frame.index] = 1
        
    # Frames not in any orientation constraint plus number of constraints.
    n_unique_orientations = (
        (n_frames - np.sum(constrained_orientation_bitmap)) +
        len(context.orientation_constraints)
    )

    # Map cameras to unique orientation sources.
    # First map unconstrained frames, then constrained ones.
    cam_to_orientation_source = np.zeros(shape=[n_frames], dtype=np.int64)
    unique_orientations = np.zeros(shape=[n_unique_orientations, 4], dtype=np.float32)
    orient_source_i = 0
    for frame in context.scene_data.frames:
        if constrained_orientation_bitmap[frame.index]:
            continue
        cam_to_orientation_source[frame.index] = orient_source_i
        unique_orientations[orient_source_i] = matrix_to_quaternion(frame.transform[:3, :3])
        orient_source_i += 1
    for constraint in context.orientation_constraints:
        source_index = orient_source_i
        orient_source_i += 1

        unique_orientations[source_index] = matrix_to_quaternion(frame.transform[:3, :3])
        for frame in constraint:
            cam_to_orientation_source[frame.index] = source_index 

    context.cam_to_orientation_source = torch.from_numpy(cam_to_orientation_source).cuda()
    context.orientation_params = torch.nn.Parameter(torch.from_numpy(unique_orientations).cuda())

def apply_colinear_constraints(context: CameraInferenceContext):
    n_frames = len(context.scene_data.frames)

    # Every co-linear constraint has one frame that is considered the parent;
    # we will create a sort of traversable tree that can resolve ancestry.
    # Along the way, we'll record the colinear direction vector that each child
    # is located along.
    frame_to_parent = -np.ones(shape=[n_frames], dtype=np.int64)
    learnable_dist_bitmap = np.zeros(shape=[n_frames], dtype=bool)
    camera_vectors = np.zeros(shape=[n_frames, 3], dtype=np.float32)
    for constraint in context.colinear_constraints:
        constraint.resolve_names(context.scene_data)
        parent_index = constraint.parent.index

        for child_data in constraint.children:

            # After resolution, all of the children in the constraint are tuples.
            child, learnable_dist = child_data

            old_parent_index = frame_to_parent[child.index]
            if old_parent_index != -1:
                raise Exception(f"Child {child.index} already has a parent: {old_parent_index}")
            
            frame_to_parent[child.index] = parent_index

            camera_vectors[child.index] = constraint.direction
            learnable_dist_bitmap[child.index] = learnable_dist
            context.learnable_vector_bitmap[child.index] = 0

    # For root frames, record translation from origin to reach the camera pose.
    for frame in context.scene_data.frames:
        parent_index = frame_to_parent[frame.index]
        if parent_index != -1:
            continue

        camera_vectors[frame.index] = frame.get_position()

    # We will resolve the root of every frame using dynamic programming DFS.
    # Meanwhile, we'll also populate the matrix whose columns indicate all the 
    # vectors used to locate a frame (ancestors + frame's unique vector).
    n_roots = 0
    frame_to_root = -np.ones(n_frames, dtype=np.int64)
    vectors_to_frames = np.zeros(shape=[n_frames, n_frames], dtype=np.float32)
    modeled_camera_translations = np.zeros(shape=[n_frames, 3], dtype=np.float32)
    dists = np.ones(shape=[n_frames], dtype=np.float32)
    for frame in context.scene_data.frames:

        # stack: tracks frames being explored.
        # root_index: set to index of the root when found.
        # parent_col_index: set to index of the matrix column of the last
        #   memoized ancestor. Only set if the root has already been explored.
        current_index = frame.index
        stack = []
        root_index = None
        parent_col_index = None
        while True:

            # Root already identified by current frame.
            if frame_to_root[current_index] != -1:
                root_index = frame_to_root[current_index]
                parent_col_index = current_index
                break

            stack.append(current_index)
            if current_index in stack[:-1]:
                raise Exception(f"Colinear constraint cycle detected: [{stack}].")

            if frame_to_parent[current_index] == -1:

                # No parent, so current frame must be the root.
                root_index = current_index
                n_roots += 1
                break
            else:

                # Recurse to identify root.
                current_index = frame_to_parent[current_index]

        # Set up column to be used for updating every frame in the stack and
        # obtain cumulative translation by any already-computed ancestors.
        if parent_col_index is not None:
            column = vectors_to_frames[:, parent_col_index].copy()
            accum_translation = modeled_camera_translations[parent_col_index]
        else:
            column = np.zeros(shape=[n_frames], dtype=np.float32)
            accum_translation = np.array([0, 0, 0])

        # Resolve roots and update matrix and other data for every frame.
        for frame_index in reversed(stack):
            frame_to_root[frame_index] = root_index
 
            if parent_col_index is None:

                # This is a root so dist scalar is 1.
                dists[frame_index] = 1
                accum_translation = camera_vectors[frame.index]
            else:

                # This is a colinear constraint child. Its dist scalar is computed
                # by projecting its prior-known position onto the colinear line
                # defined by its parent.
                prior_cam = frame.get_position()
                transform = context.get_orientation(parent_col_index).detach().cpu().numpy()
                direction = camera_vectors[frame_index] @ transform
                dist = np.dot(prior_cam - accum_translation, direction)
                dists[frame_index] = dist
                accum_translation += dist * direction
            modeled_camera_translations[frame_index] = accum_translation

            # Update the matrix. Each column specifies to use the frame's own
            # vector and the vectors of any and all ancestors.
            column[frame_index] = 1
            vectors_to_frames[:, frame_index] = column

            parent_col_index = frame_index
    assert(-1 not in frame_to_root)

    # Set up vector and dist sources.
    # Vectors can either be fixed, learnable or derived from parent frame (colinear).
    # Dists can either be fixed or learnable.
    n_learnable_roots = np.sum(context.learnable_vector_bitmap).astype(np.int64)
    n_colinear_vectors = n_frames - n_roots
    n_learnable_dists = np.sum(learnable_dist_bitmap).astype(np.int64)
    context.vector_params = torch.nn.Parameter(torch.zeros(size=[n_learnable_roots, 3]).cuda(), requires_grad=True)
    context.dist_params = torch.nn.Parameter(torch.zeros(size=[n_learnable_dists]).cuda(), requires_grad=True)
    context.learnable_vectors = torch.zeros(size=[n_learnable_roots], dtype=torch.int64)
    context.colinear_vectors = torch.zeros(size=[n_colinear_vectors, 2], dtype=torch.int64)
    context.learnable_dists = torch.zeros(size=[n_learnable_dists], dtype=torch.int64)
    context.vectors_to_frame_matrix = torch.from_numpy(vectors_to_frames).cuda()

    # Assign learnable params and colinear vector sources.
    learnable_vector_i = 0
    colinear_vector_i = 0
    learnable_dist_i = 0
    for (i, frame) in enumerate(context.scene_data.frames):
        
        if context.learnable_vector_bitmap[i]:
            context.learnable_vectors[learnable_vector_i] = i 
            context.vector_params.data[learnable_vector_i] = torch.from_numpy(camera_vectors[i]).cuda() 
            learnable_vector_i += 1
        elif frame_to_parent[i] != -1:
            parent_index = frame_to_parent[i]
            context.colinear_vectors[colinear_vector_i] = torch.tensor([i, parent_index]).cuda()
            colinear_vector_i += 1

        if learnable_dist_bitmap[i]:
            context.learnable_dists[learnable_dist_i] = i
            context.dist_params.data[learnable_dist_i] = float(dists[i])
            learnable_dist_i += 1

    # These are the default values for the camera vectors and 
    # vector dist tensors. For each solver iteration, these tensors are copied
    # and values that aren't fixed are overridden.
    context.default_camera_vectors = torch.from_numpy(camera_vectors).cuda()
    context.default_dists = torch.from_numpy(dists).cuda()

    pass


def _infer_zparams(scene: RawSceneData):
    n_frames = len(scene.frames)

    fl_x = scene.proj_params.fl_x
    fl_y = scene.proj_params.fl_y
    w = scene.proj_params.w
    h = scene.proj_params.h

    shape = scene.frames[0].get_depth().shape

    inv_proj = _inverse_projection_matrix(w, h, fl_x, fl_y, 
            torch.tensor(0.1, dtype=torch.float32), 
            torch.tensor(1000,  dtype=torch.float32)).T.cuda()

    f1 = scene.lookup('003')
    f2 = scene.lookup('006')
    d1 = f1.get_depth()[720, 1280, 0]
    d2 = f2.get_depth()[720, 1280, 0]
    dist = np.linalg.norm(f1.get_position() - f2.get_position())
    slope = dist / np.abs(d2 - d1)

    f1 = scene.lookup('004')
    f2 = scene.lookup('005')
    d1 = f1.get_depth()[720, 1280, 0]
    d2 = f2.get_depth()[720, 1280, 0]
    dist = np.linalg.norm(f1.get_position() - f2.get_position())
    slope2 = dist / np.abs(d2 - d1)

    fixed_pos_constraints = [
        '003'
    ]
    orientation_constraints = [
       # ['003', '004', '005', '006']
    ]
    colinear_constraints = [
       #    ColinearConstraint('003', (0, 0, 1), [('006',  False)])
    ]
    context = CameraInferenceContext(
        scene, 
        fixed_pos_constraints,                 
        orientation_constraints, 
        colinear_constraints
    )
    apply_fixed_pos_constraints(context)
    apply_orientation_constraints(context)
    apply_colinear_constraints(context)

    #scene.feature_data.filter_by_err(10)

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

    def compute_transforms():

        # Set up per-camera transforms.
        total_transforms = torch.zeros(size=[n_frames, 4, 4]).cuda()
        total_transforms[:, :3, :3] = make_rotation(context.orientation_params[context.cam_to_orientation_source])
        total_transforms[:, 3, 3] = 1

        # Translations use matrix multiplication to factor in colinear constraints.
        dists = context.default_dists.clone()
        dists[context.learnable_dists] = context.dist_params
        camera_vectors = context.default_camera_vectors.clone()
        camera_vectors[context.learnable_vectors] = context.vector_params
        colinear_indices = context.colinear_vectors[:, 0]
        colinear_source_cams = context.colinear_vectors[:, 1]
        if len(colinear_indices) > 0:
            camera_vectors[colinear_indices] = (context.default_camera_vectors[colinear_indices].view([-1, 1, 3]) @ total_transforms[colinear_source_cams, :3, :3])[0]
        total_transforms[:, 3, 0] = (dists.view([1, -1]) * camera_vectors[:, 0].view([1, -1])) @ context.vectors_to_frame_matrix
        total_transforms[:, 3, 1] = (dists.view([1, -1]) * camera_vectors[:, 1].view([1, -1])) @ context.vectors_to_frame_matrix
        total_transforms[:, 3, 2] = (dists.view([1, -1]) * camera_vectors[:, 2].view([1, -1])) @ context.vectors_to_frame_matrix

        return total_transforms

    def objective_fn():
        optimizer.zero_grad()

        zstats = torch.exp(zstats_param)

        total_transforms = compute_transforms()

        inv_view = total_transforms[frame_feature_to_cam]

        coords_3D, _, _ = _depth_to_3D(frame_feature_depths, inv_proj, inv_view, 
                    zstats[0], slope + zstats[0], pixel_coords_2D=frame_feature_pixel_coords, 
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
    
    #optimizer = torch.optim.Adam([zstats_param, context.orientation_params, context.vector_params, context.dist_params], lr=2e-1)
    optimizer = torch.optim.Adam([zstats_param], lr=2e-1)

    '''cam_param.requires_grad = False
    cam_orientation_param.requires_grad = False

    for t in range(0, 100):

        if t == 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 2e-2

        loss = optimizer.step(closure=objective_fn)
        print(f"Step {t}: ", loss)

    cam_param.requires_grad = True
    cam_orientation_param.requires_grad = True'''
    for t in range(0, 100):

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

    zstats = torch.exp(zstats_param)
    scene.proj_params.znear = float(zstats[0].detach())
    scene.proj_params.zfar = slope + scene.proj_params.znear 

    scene.proj_params.znear = 0.07
    scene.proj_params.zfar = slope + scene.proj_params.zfar
    print(f"Selected znear and zfar: {scene.proj_params.znear} and {scene.proj_params.zfar}")
    
    # Permanently apply our adjusted camera positions.
    #for (cam_idx, frame) in enumerate(scene.frames):
    #    frame.transform[:3,:3] = np.matmul(frame.transform[:3:,:3], make_rotation(cam_orientation_param[cam_idx].unsqueeze(0)).detach().cpu().numpy()) 
    #    frame.transform[3:,:3] = frame.transform[3:,:3] + cam_param[cam_idx].detach().cpu().numpy()

    transforms = compute_transforms()
    #for (cam_idx, frame) in enumerate(scene.frames):
    #    frame.transform = transforms[cam_idx].detach().cpu().numpy()

    #raise Exception("TODO: Apply params to transforms.")
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

    if uses_depth_map:
        depths = depth_data[pixel_coords_2D[:,1], pixel_coords_2D[:,0]][:,0]
    else:
        depths = depth_data
    #depths = torch.minimum(depths, torch.tensor(0.999).cuda())

    # Create homogenous normalized coordinates to match each depth map pixel.
    # Shape them into [B, 4].
    x_coords = 2 * pixel_coords_2D[:,0] / shape[1] - 1 + 1 / shape[1]
    y_coords = 2 * pixel_coords_2D[:,1] / shape[0] - 1 + 1 / shape[0]
    z_coords = 2 * depths - 1
    projected_coords = torch.stack([x_coords, y_coords, z_coords, torch.ones_like(x_coords, dtype=torch.float32)], axis=-1)

    # Calculate the view space coordinates
    view_coordinates = projected_coords @ inv_projection
    view_coordinates = view_coordinates / view_coordinates[:,3].reshape([B, 1])
    #view_coordinates = view_coordinates / torch.linalg.norm(view_coordinates[:,:3], dim=1).unsqueeze(-1)
    #z_want = znear + depths * (zfar - znear)
    #z_want = -z_want
    #view_coordinates = view_coordinates * (z_want / view_coordinates[:,2]).unsqueeze(-1)
    #view_coordinates[:,3] = 1

    # Calculate the world space coordinates
    if len(inv_view.shape) == 2:
        point_homogeneous_world = view_coordinates @ inv_view
    else:
        point_homogeneous_world = view_coordinates.reshape([-1, 1, 4]) @ inv_view
        point_homogeneous_world = point_homogeneous_world.reshape([-1, 4])

    # Normalize to get the 3D coordinates
    point_3D = point_homogeneous_world[:,:3] / point_homogeneous_world[:,3].reshape([B, 1])

    limit = 30
    dists = torch.linalg.norm(point_3D, dim=1)
    mask = dists >= limit
    divisor = dists / limit * mask + ~mask * 1
    point_3D = point_3D / divisor.unsqueeze(-1)

    if colour_map is not None:
        colours = colour_map.reshape([shape[0] * shape[1], 3])
    else:
        colours = depths

    #tex = torch.stack((x_coords * 0.5 + 0.5, y_coords * 0.5 + 0.5, flat_depth_map), dim=1)

    return point_3D, colours, None


def _perspective_projection_matrix(width, height, camera_angle_x, camera_angle_y, znear, zfar):

    aspect_ratio = width / height
    a = 1 / (torch.tan(torch.tensor(0.5 * camera_angle_x)))
    b = 1 / (torch.tan(torch.tensor(0.5 * camera_angle_y)))

    projection_matrix = torch.tensor([
        a, 0, 0, 0,
        0, b, 0, 0,
        0, 0, -(zfar + znear) / (zfar - znear), -(2 * zfar * znear) / (zfar - znear),
        0, 0, -1, 0
    ]).reshape((4, 4))

    return projection_matrix


def _inverse_projection_matrix(width, height, camera_angle_x, camera_angle_y, znear, zfar):
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

    return inv_projection_matrix


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



def matrix_to_quaternion(matrix):

    # Ensure the matrix is a valid 3x3 rotation matrix
    if matrix.shape != (3, 3):
        raise ValueError("Input matrix must be a 3x3 matrix")

    # Ensure the matrix is a proper rotation matrix
    is_rotation_matrix = np.allclose(np.dot(matrix, matrix.T), np.identity(3))
    if not is_rotation_matrix:
        raise ValueError("Input matrix is not a valid rotation matrix")

    # Extract the trace and diagonal elements
    trace = np.trace(matrix)
    diag = np.diagonal(matrix)

    # Calculate the quaternion components
    w = np.sqrt(1.0 + trace) / 2.0
    x = (matrix[2, 1] - matrix[1, 2]) / (4.0 * w)
    y = (matrix[0, 2] - matrix[2, 0]) / (4.0 * w)
    z = (matrix[1, 0] - matrix[0, 1]) / (4.0 * w)

    # Ensure the quaternion is in the correct order (x, y, z, w)
    quaternion = np.array([x, y, z, w])

    return quaternion