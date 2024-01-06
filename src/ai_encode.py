''' Encode the scene with AI operators. '''

from typing import Set, List
import numpy as np
import torch
from .source_loader import RawSceneData
from .projection import unproject, unproject_ray


N_BITS_PER_BLOCK_DIM = 20

MAX_DIST = 30


def run(scene_data: RawSceneData):
    ''' Run the ai-encoder on the given scene. '''
    
    filled_block_indices, minimum, maximum = _identify_filled_blocks(scene_data=scene_data)
    tree, blocks = _construct_octree(filled_block_indices, minimum, maximum)
    #_test_octree(scene_data, tree, blocks)
    _collect_data(scene_data, tree, blocks, len(filled_block_indices))
    pass


class Octree:
    ''' Divides space recursively to enable finding nearby blocks faster. '''

    def __init__(self, nodes, n_levels, minimum):
        ''' Constructs the tree.
        
            nodes: [N, 8] array of N nodes. Each node consists of 8 child 
                indices or -1 if no child.
            n_levels: number of levels of the tree BUT excludes the leaf level.
            minimum: minimum (most negative) physical coordinates covered by 
                the tree. 
        '''

        self.nodes = nodes
        self.n_levels = n_levels
        self.minimum = minimum

    def to_torch_cuda(self):
        ''' Turns the octree into a torch cuda tensor. '''
        if not isinstance(self.nodes, torch.Tensor):
            self.nodes = torch.from_numpy(self.nodes).cuda()
        if not isinstance(self.minimum, torch.Tensor):
            self.minimum = torch.from_numpy(self.minimum).cuda()


class Block:
    ''' A block that can contain geometry and is effectively a leaf of the
        Octree. '''
    
    def __init__(self, min_coords):
        self.min_coords = min_coords


def _identify_filled_blocks(scene_data: RawSceneData):
    ''' Identifies each block in 3D space that contains at least one surface
        sample. '''

    filled_block_indices = torch.zeros(size=[0], dtype=torch.int64).cuda()
    minimum = None
    maximum = None
    for frame in scene_data.frames:

        # Find blocks that each world point would belong in.
        world_points = unproject(scene_data, frame, "cuda", max_dist=MAX_DIST)
        block_coords = torch.floor(world_points).to(torch.int64)
        this_min, _ = torch.min(block_coords, dim=0)
        this_max, _ = torch.max(block_coords, dim=0)
        if minimum is None:
            minimum = this_min
            maximum = this_max
        else:
            minimum = torch.minimum(minimum, this_min)
            maximum = torch.maximum(minimum, this_max)
        block_indices = _compress_block_indices(block_coords)

        # Add to our knowledge set of filled blocks.
        uniques = torch.unique(block_indices)
        filled_block_indices = torch.concat((filled_block_indices, uniques))
        filled_block_indices = torch.unique(filled_block_indices)
    
    return filled_block_indices, minimum, maximum


def _construct_octree(filled_block_indices, minimum, maximum) -> Octree:
    ''' Construct an octree using the indices of the filled blocks. '''

    # We'll use numpy for this step instead of torch.
    filled_block_indices_np = filled_block_indices.cpu().numpy()
    minimum_np = minimum.cpu().numpy()
    maximum_np = maximum.cpu().numpy()

    # Create tree with root. n_levels excludes leaves.
    greatest_range = (maximum_np - minimum_np).max()
    n_levels = np.ceil(np.log2(greatest_range)).astype(np.int64)
    tree = -np.ones(shape=[1, 8], dtype=np.int64)

    # Decode block indices.
    coords = _decompress_block_indices(filled_block_indices_np)

    n_blocks = len(filled_block_indices_np)
    blocks = [Block(coords[i].copy()) for i in range(0, n_blocks)]

    # Add each block to the tree one at a time.
    coords = coords - minimum_np
    for i in range(0, n_blocks):

        # Note level here ranges in [1, n_levels]. Since leaves are excluded,
        # this corresponds to the actual logarithmic physical size of the level.
        relative_coords = coords[i]
        node_index = 0
        for level in range(n_levels - 1, -1, -1):

            # Find which of the 8 children the block is in.
            mid = 1 << level
            side_mask = relative_coords >= mid
            relative_child = side_mask[0] + (side_mask[1] << 1) + (side_mask[2] << 2)

            child_index = tree[node_index, relative_child]
            if level == 0:

                # Add block to the tree as a leaf node.
                assert(child_index) == -1
                tree[node_index, relative_child] = i
            
            elif child_index == -1:

                # Child node does not exist already; create it.
                child_index = len(tree)
                child_node = -np.ones(shape=[1, 8], dtype=np.int64)
                tree = np.concatenate((tree, child_node), axis=0)
                tree[node_index, relative_child] = child_index

            relative_coords[0] -= side_mask[0] << level
            relative_coords[1] -= side_mask[1] << level
            relative_coords[2] -= side_mask[2] << level
            node_index = child_index

    tree = Octree(tree, n_levels, minimum)
    tree.to_torch_cuda()
    return tree, blocks


def _collect_data(scene_data: RawSceneData, tree: Octree, blocks: List,
        n_blocks: int):
    ''' Raytrace through the scene from each frame at a time to identify
        boundaries and known outside voxels. '''
    
    # TODO: If too many blocks, we'll need to apply some kind of LRU cache
    # strategy and write to disk.
    for frame in scene_data.frames:

        all_ray_starts, all_ray_dirs = unproject_ray(scene_data, frame)

        # Keeps track of what rays are still traversing.
        n_rays = 1 #len(all_ray_starts)
        remaining_ray_indices = torch.arange(0, n_rays, dtype=torch.int64).cuda()

        # Ray state variables. These are permanently allocated, one per ray.
        # ray_stacks: set of stack frames containing node index and current 
        #   child (0-7) being explored.
        # ray_stack_heads: current stack frame.
        # ray_pos_stacks: minimum position of each octree cell in the stack
        #   history.
        ray_stacks = torch.zeros([n_rays, tree.n_levels + 1, 2], dtype=torch.int64).cuda()
        ray_stack_heads = torch.zeros([n_rays], dtype=torch.int64).cuda()
        ray_pos_stacks = torch.zeros([n_rays, tree.n_levels + 1, 3], dtype=torch.float).cuda()
        ray_pos_stacks[:,0] = tree.minimum.reshape([1, 3])

        # Loop until all rays done.
        n_visits = 0
        visit_set = set()
        while True:

            # Run tree-traversal tracer algorithm until every ray hits a block or
            # exits the tree.
            tree_traversal_ray_indices = remaining_ray_indices.clone()
            while len(tree_traversal_ray_indices) > 0:
                
                # This algorithm explores one child of the current node per 
                # iteration.

                # Exit leaf frame if in one.
                leaf_mask = ray_stack_heads[tree_traversal_ray_indices] == tree.n_levels
                ray_stack_heads[tree_traversal_ray_indices[leaf_mask]] -= 1
                if leaf_mask.sum() > 0:
                    n_visits += 1
                    #print(ray_stacks[leaf_mask, -1, 0])
                    visit_set.add(int(ray_stacks[0, -1, 0]))
                del leaf_mask

                # Exit any stack frames if at the end of them.
                check_end_indices = tree_traversal_ray_indices
                while len(check_end_indices) > 0:
                    heads =  ray_stack_heads[check_end_indices]
                    mask = ray_stacks[check_end_indices, heads, 1] == 8
                    end_frames = check_end_indices[mask]
                    ray_stack_heads[end_frames] -= 1
                    check_end_indices = end_frames 
                    if len(end_frames) > 0:
                        print("Unwind")
                        pass

                # Check finish condition.
                still_going_mask = ray_stack_heads[tree_traversal_ray_indices] > -1
                tree_traversal_ray_indices = tree_traversal_ray_indices[still_going_mask]
                still_going_mask = ray_stack_heads[remaining_ray_indices] > -1
                remaining_ray_indices = remaining_ray_indices[still_going_mask]
                del still_going_mask
                if len(tree_traversal_ray_indices) == 0:
                    break

                # Identify child node to explore.
                stack_locs = ray_stacks[
                    tree_traversal_ray_indices, 
                    ray_stack_heads[tree_traversal_ray_indices]
                ]
                parent_node_indices = stack_locs[:, 0]
                child_node_indices = tree.nodes[parent_node_indices, stack_locs[:, 1]]

                # Check if child node is not null and that the ray intersects
                # it.
                child_exists_mask = (child_node_indices != -1)
                exist_child_ray_indices = tree_traversal_ray_indices[child_exists_mask]
                if len(exist_child_ray_indices) > 0:
                    stack_heads = ray_stack_heads[exist_child_ray_indices]
                    parent_minimums = ray_pos_stacks[
                        exist_child_ray_indices,
                        stack_heads
                    ]
                    print("Parent minimums:", parent_minimums[0].cpu().numpy())
                    levels = tree.n_levels - stack_heads - 1 # TODO: Verify.
                    child_numbers = ray_stacks[
                        exist_child_ray_indices,
                        stack_heads,
                        1
                    ]
                    child_minimums = parent_minimums + _octree_child_to_coords(child_numbers, levels)
                    child_maximums = child_minimums + (1 << levels)

                    # For use later on, restrict child minimums to only show 
                    # the minimums of rays that will actually be entered.
                    intersection_mask = _ray_box_intersect(
                            all_ray_starts[exist_child_ray_indices],
                            all_ray_dirs[exist_child_ray_indices],
                            box_mins=child_minimums,
                            box_maxes=child_maximums
                    )
                    child_minimums = child_minimums[intersection_mask]
                    enter_ray_indices = exist_child_ray_indices[intersection_mask]
                else:
                    print("Skipping child")
                    child_minimums = None
                    enter_ray_indices = exist_child_ray_indices

                # Conditionally enter child node if not null and the ray
                # intersects it.
                # 1. Advance stack head.
                # 2. Write the new stack frame.
                old_stack_heads = ray_stack_heads[tree_traversal_ray_indices]
                if len(enter_ray_indices) > 0:
                    print("Entering child: ", child_minimums[0].cpu().numpy())
                    ray_stack_heads[enter_ray_indices] += 1
                    stack_heads = ray_stack_heads[enter_ray_indices]
                    ray_stacks[
                        enter_ray_indices, 
                        stack_heads,
                        0
                    ] = child_node_indices
                    ray_stacks[
                        enter_ray_indices, 
                        stack_heads,
                        1
                    ] = 0
                    ray_pos_stacks[
                       enter_ray_indices,
                       stack_heads
                    ] = child_minimums

                # Advance the child iterator in the current/parent stack frame. 
                # This can overflow, but that would be handled at the start of
                # the next iteration.
                ray_stacks[
                    tree_traversal_ray_indices, 
                    old_stack_heads,
                    1
                ] += 1

                # Remove nodes from this traversal that have reached a leaf.
                internal_mask = ray_stack_heads[tree_traversal_ray_indices] != tree.n_levels
                tree_traversal_ray_indices = tree_traversal_ray_indices[internal_mask]
                del internal_mask 

            print("Leaf exploration")
            if len(remaining_ray_indices) == 0:
                break
            
            # Now run the the block traversal algorithm until every ray exits the
            # block.
            #block_traversal_ray_indices = remaining_ray_indices.clone()
            #while len(block_traversal_ray_indices) > 0:
            pass


        # Coalesce writes to the same blocks together.
        pass


def _octree_child_to_coords(child_numbers, log_level):
    ''' Takes a tensor full of octree child numbers (0-7) and turns it into a 
        tensor full of relative coordinates. '''

    x = child_numbers % 2
    y = (child_numbers % 4) >= 2
    z = child_numbers >= 4
    coords = torch.stack((x, y, z), dim=1) << log_level
    return coords


def _ray_box_intersect(ray_origins, ray_directions, box_mins, box_maxes):

    div0_mask = torch.abs(ray_directions) < 1e-8
    inv_dir = 1 / (div0_mask * 1e-8 + ray_directions)
    
    t_mins = (box_mins - ray_origins) * inv_dir
    t_maxes = (box_maxes - ray_origins) * inv_dir

    tmin = torch.max(torch.minimum(t_mins, t_maxes), dim=1).values
    tmax = torch.min(torch.maximum(t_mins, t_maxes), dim=1).values
    intersected = (tmax > tmin) & (tmax > 0)
    return intersected


def _compress_block_indices(block_coords):

    # Express block coordinates as a single spatial index value. 
    # We add (1 << (B - 1)) to prevent negatives.
    B = N_BITS_PER_BLOCK_DIM
    block_coords = block_coords + (1 << (B - 1))
    block_indices = block_coords[:, 0] + (block_coords[:, 1] << B) + (block_coords[:, 2] << (2 * B))
    return block_indices


def _decompress_block_indices(filled_block_indices_np):
    ''' Decompress the given compressed block indices into (x, y, z) block
        coordinates. '''

    B = N_BITS_PER_BLOCK_DIM
    x_mask = (1 << B) - 1
    y_mask = ((1 << B) - 1) << B
    z_mask = ((1 << B) - 1) << (2 * B)
    filled_block_indices_np = filled_block_indices_np.reshape([-1, 1])
    x = (filled_block_indices_np & x_mask)
    y = ((filled_block_indices_np & y_mask) >> B)
    z = ((filled_block_indices_np & z_mask) >> (2 * B))
    x = x - (1 << (B - 1))
    y = y - (1 << (B - 1))
    z = z - (1 << (B - 1))
    return np.concatenate((x, y, z), axis=1)


def _test_octree(scene_data: RawSceneData, tree: Octree, blocks: List):
    ''' Test that the octree was constructed properly. '''
    
    # Create a string-based set of all blocks.
    all_uniques = np.zeros(shape=[0], dtype=np.int64)
    for frame in scene_data.frames:
        print(f"Adding {frame.colour_path} to hashset.")
        world_points = unproject(scene_data, frame, max_dist=MAX_DIST)
        block_coords = torch.floor(world_points).to(torch.int64)
        block_indices = _compress_block_indices(block_coords)
        uniques = np.unique(block_indices.cpu())
        all_uniques = np.concatenate((all_uniques, uniques))
    
    # Explore octree to verify all leaves match the set and that the right
    # number of leafs (blocks) are registered in the octree.
    print("Exploring octree.")
    all_uniques = set(all_uniques)
    n_octree_leaves = _test_explore_octree(tree, blocks, all_uniques, 0, 
            tree.n_levels - 1, np.array(tree.minimum.cpu()))
    assert(n_octree_leaves == len(all_uniques)) 

    
def _test_explore_octree(tree: Octree, blocks, coord_set: Set, index: int, 
        log_level: int, coords) -> int:
    
    if log_level == -1:

        # Leaf level.
        encoded_coords = _compress_block_indices(coords.reshape([1, -1]))[0]
        encoded_coords = int(encoded_coords)
        assert(encoded_coords in coord_set)
        assert(np.all(blocks[index].min_coords == coords))
        return 1

    n_leaves = 0
    for (index, child) in enumerate(tree.nodes[index]):
        pos_offset = np.array([index % 2, (index % 4) >= 2, (index >= 4)]) << log_level
        if child != -1:
            n_leaves += _test_explore_octree(tree, blocks, coord_set, child, 
                    log_level - 1, coords + pos_offset)
    
    return n_leaves
            


    





    