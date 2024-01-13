''' Encode the scene with AI operators. '''

from typing import Set, List
import numpy as np
import torch
from .source_loader import RawSceneData
from .projection import unproject, unproject_ray


N_BITS_PER_BLOCK_DIM = 20

MAX_DIST = 30

# Number of grid cells that subdivide a block.
BLOCK_GRID_SIZE = 32

# Number of blocks to process in one iteration of ray-block processing.
N_BLOCKS_PER_IT = 64


def run(scene_data: RawSceneData):
    ''' Run the ai-encoder on the given scene. '''
    
    filled_block_indices, minimum, maximum = _identify_filled_blocks(scene_data=scene_data)
    tree, blocks, block_starts = _construct_octree(filled_block_indices, minimum, maximum)
    #_test_octree(scene_data, tree, blocks)
    _collect_data(scene_data, tree, blocks, block_starts, len(filled_block_indices))
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


class BlockDataCache:
    def __init__(self, n_blocks, n_cells, device):
        self.n_blocks = n_blocks
        self.n_cells = n_cells
        self.clock = 0
        self.device_type = device

        # All cells currently free to accept block data.
        self.free_cells = torch.arange(n_cells, dtype=torch.int64, device=device)
        
        # Mapping of overall block index to cell index.
        self.block_to_cell = -1 * torch.ones(size=[n_blocks], dtype=torch.int64, device=device)

        # Mapping of cells to the block they contain. 
        self.cell_to_block = -1 * torch.ones(size=[n_cells], dtype=torch.int64, device=device)

        # Clock values for cells to determine eviction order.
        self.cell_clocks = torch.zeros(size=[n_cells], dtype=torch.int64, device=device)

        # Occupancy data for blocks.
        self.occupancy = torch.zeros(
                size=[n_cells, BLOCK_GRID_SIZE, BLOCK_GRID_SIZE, BLOCK_GRID_SIZE],
                dtype=bool,
                device=device
        )

        # Actual hit datapoints.


    def load(self, block_indices, occupancy):
        ''' Load block data into the cache.
            Precondition: indicated blocks are not already in the cache.
            Precondition: there are enough free blocks to load the data into;
                this will not evict anything.
         
            block_indices: indices of the blocks to load. [N]
            occupancy: occupancy data for the blocks. [N]
        '''
        assert(len(block_indices) <= self.n_cells)
        assert(len(block_indices) <= len(self.free_cells))

        block_indices = block_indices.to(self.device_type)
        occupancy = occupancy.to(self.device_type)

        cell_indices = self.free_cells[:len(block_indices)]
        self.free_cells = self.free_cells[:len(block_indices)]

        self.block_to_cell[block_indices] = cell_indices
        self.cell_to_block[cell_indices] = block_indices
        self.occupancy[cell_indices] = occupancy

        # Mark that the cell has been accessed this clock cycle.
        self.mark(cell_indices)
        
    def mark(self, cell_indices):
        ''' Mark that cells have been accessed this clock cycle. '''
        self.cell_clocks[cell_indices] = self.clock
        self.clock += 1

    def evict(self, cell_indices):
        ''' Evict indicated cells, returning their data. '''

        # This can be used both just to move data and for eviction.

        # Precondition: Ensure cell_indices are all actually allocated.
        # Load evicted cell data into tensors/list.
        # Mark allocated mask as free there.
        # Add cell_indices to free_cells. 
        # Return the evicted data. Don't change device yet.
        block_indices, occupancy = (
                self.cell_to_block[cell_indices].clone(), 
                self.occupancy[cell_indices].clone()
        )

        self.cell_to_block[cell_indices] = -1
        self.block_to_cell[block_indices] = -1
        self.free_cells = torch.concat((self.free_cells, cell_indices))

        return block_indices, occupancy

    def evict_n(self, n_cells):
        ''' Evict the indicated number of cells, returning their data. '''

        # Identify allocated cells.
        allocated_cells = torch.where(self.cell_to_block != -1)

        # Sort allocated cells by clock and choose the n oldest ones.
        sort_order = torch.argsort(self.cell_clocks[allocated_cells])
        evicted_cells = allocated_cells[sort_order][:n_cells]

        return self.evict(evicted_cells)


class BlockDataCacheSystem:
    ''' Manages block data in a cache-like pattern to avoid out-of-memory on 
        device and cpu.
         
        In this style of cache:
        - A block is only loaded into one of either device or cpu at a time.
        - Data can be loaded directly into device from disc.    
    '''
    
    def __init__(self, n_total_blocks, n_cpu_blocks, n_device_blocks,
            device_type="cuda"):
        self.n_total_blocks = n_total_blocks
        self.device_type = device_type
        self.cpu_cache = BlockDataCache(n_total_blocks, n_cpu_blocks, device="cpu")
        self.device_cache = BlockDataCache(n_total_blocks, n_device_blocks, device=device_type)

    def ensure_loaded(self, block_indices):
        ''' Ensures all the given blocks are resident in device cache. '''
        assert(len(block_indices) <= self.device_cache.n_cells)

        # Identify blocks that have not been loaded and then load them.
        cache = self.device_cache
        missing_mask = cache.block_to_cell[block_indices] == -1
        missing_blocks = block_indices[missing_mask]
        
        # Load missing from cpu.
        missing_blocks = missing_blocks.to("cpu")
        cpu_cells = self.cpu_cache.block_to_cell[missing_blocks]
        found_block_mask = cpu_cells != -1
        found_blocks = missing_blocks[found_block_mask]
        found_cells = cpu_cells[found_blocks]
        if len(found_cells) > 0:
            self.load('device', *self.cpu_cache.evict(found_cells))
            if len(found_cells) == len(missing_blocks):
                return
        missing_blocks = missing_blocks[~found_block_mask]
        del cpu_cells, found_block_mask, found_blocks, found_cells

        # Load remaining from disc if they exist.
        # TODO: Check disc.

        # Otherwise, create new data for missing blocks.
        new_occupancy = torch.zeros(
                size=[len(missing_blocks), BLOCK_GRID_SIZE, BLOCK_GRID_SIZE, BLOCK_GRID_SIZE],
                dtype=bool,
                device=self.device_type
        )
        self.load('device', missing_blocks, new_occupancy)
        
    def load(self, cache_type, block_indices, occupancy):
        ''' Load data into the indicated cache.
        
            Precondition: the given blocks are missing from the cache.
        '''

        cache = self.cpu_cache if cache_type == 'cpu' else self.device_cache
        assert(len(block_indices) <= cache.n_cells)

        # Get free cells. If not enough, we must evict.
        n_free_cells = len(cache.free_cells)
        if n_free_cells < len(block_indices):
            evicted_data = cache.evict_n(len(block_indices) - n_free_cells)
        else:
            evicted_data = None
        
        # Now that we've made room, insert the data into the cache.
        cache.load(block_indices, occupancy)

        # Deal with data evicted from the cache to make room before.
        if evicted_data is not None and cache_type == 'device':
            self.load('cpu', *evicted_data)
        elif evicted_data is not None: 
            raise Exception("Eviction to make room in CPU is not implemented.")


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
    block_starts = torch.from_numpy(coords).cuda()

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
    return tree, blocks, block_starts


def _collect_data(scene_data: RawSceneData, tree: Octree, blocks: List,
        block_starts, n_blocks: int):
    ''' Raytrace through the scene from each frame at a time to identify
        boundaries and known outside voxels. '''

    # Note that this algorithm does not necessarily traverse blocks in a set 
    # order.

    cache = BlockDataCacheSystem(n_blocks, n_blocks, 128)

    # TODO: If too many blocks, we'll need to apply some kind of LRU cache
    # strategy and write to disk.
    for frame in scene_data.frames:

        all_ray_starts, all_ray_dirs, all_ray_maxes = unproject_ray(scene_data, frame)

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
        all_ray_ts = torch.zeros([n_rays], dtype=torch.float32).cuda()

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
                    levels = tree.n_levels - stack_heads - 1
                    child_numbers = ray_stacks[
                        exist_child_ray_indices,
                        stack_heads,
                        1
                    ]
                    child_minimums = parent_minimums + _octree_child_to_coords(child_numbers, levels)
                    child_maximums = child_minimums + (1 << levels)

                    # For use later on, restrict child minimums to only show 
                    # the minimums of rays that will actually be entered.
                    intersection_mask, ray_mins_t, ray_maxes_t = _ray_box_intersect(
                            all_ray_starts[exist_child_ray_indices],
                            all_ray_dirs[exist_child_ray_indices],
                            all_ray_maxes[exist_child_ray_indices],
                            box_mins=child_minimums,
                            box_maxes=child_maximums
                    )
                    child_minimums = child_minimums[intersection_mask]
                    enter_ray_indices = exist_child_ray_indices[intersection_mask]

                    # Report t of first hit (must have actually intersected AND be positive).
                    pos_t_mask = ray_mins_t[intersection_mask] >= 0
                    all_ray_ts[enter_ray_indices] = (pos_t_mask) * ray_mins_t[intersection_mask] + ~pos_t_mask * ray_maxes_t[intersection_mask]

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
            
            # Run block traversal.

            # First we identify where each ray begins relative to its block.
            block_traversal_ray_indices = remaining_ray_indices.clone()
            ray_block_indices = ray_stacks[:, -1, 0]

            # Process blocks in groups (ordered by index).
            unique_blocks = torch.unique(ray_block_indices)
            n_hit_blocks = len(unique_blocks)
            for i in range(0, n_hit_blocks, N_BLOCKS_PER_IT):

                first_block_index = i
                last_block_index = min(i + N_BLOCKS_PER_IT - 1, n_hit_blocks - 1)
                cache.ensure_loaded(unique_blocks[first_block_index:last_block_index + 1])

                # Pay attention only to rays that hit this group of blocks.
                first_block = unique_blocks[first_block_index]
                last_block = unique_blocks[last_block_index]
                accept_mask = first_block <= ray_block_indices & ray_block_indices <= last_block
                rays_this_it = block_traversal_ray_indices[accept_mask]

                while True:

                    # Compute ray head position.
                    ray_heads = all_ray_starts[rays_this_it] + all_ray_dirs[rays_this_it] * (all_ray_ts[rays_this_it] + 1e-5)
                    relative_ray_pos = ray_heads - block_starts[ray_block_indices[rays_this_it]]

                    # Rays done when past maximum or out of cell bounds. 
                    # Remember to delete entries for aux arrays.
                    continue_mask = (
                            (all_ray_ts[rays_this_it] <= all_ray_maxes[rays_this_it]) | 
                            torch.all(torch.abs(relative_ray_pos) < 1, dim=1)
                    )
                    rays_this_it = rays_this_it[continue_mask]
                    if len(rays_this_it) == 0:
                        break
                    ray_heads = ray_heads[continue_mask]
                    relative_ray_pos = relative_ray_pos[continue_mask]

                    # Mark current grid cell as having an open section.
                    dest_block_indices = cache.device_cache.block_to_cell[ray_block_indices[rays_this_it]]
                    grid_cells = torch.floor(relative_ray_pos * BLOCK_GRID_SIZE).to(torch.int64)
                    cache.device_cache.occupancy[dest_block_indices, grid_cells[:, 2], grid_cells[:, 1], grid_cells[:, 0]] = True

                    # Advance rays into the next grid cell by casting against 
                    # the boundaries of the current cell.
                    min_bounds = block_starts[ray_block_indices[rays_this_it]] + grid_cells / BLOCK_GRID_SIZE
                    max_bounds = min_bounds + 1 / BLOCK_GRID_SIZE
                    intersection_mask, _, ray_maxes_t = _ray_box_intersect(
                            ray_origins=all_ray_starts[rays_this_it], 
                            ray_directions=all_ray_dirs[rays_this_it],
                            ray_maxes=all_ray_maxes[rays_this_it],
                            box_mins=min_bounds,
                            box_maxes=max_bounds
                    )
                    all_ray_ts[rays_this_it] = ray_maxes_t + 1e-5


def _octree_child_to_coords(child_numbers, log_level):
    ''' Takes a tensor full of octree child numbers (0-7) and turns it into a 
        tensor full of relative coordinates. '''

    x = child_numbers % 2
    y = (child_numbers % 4) >= 2
    z = child_numbers >= 4
    coords = torch.stack((x, y, z), dim=1) << log_level
    return coords


def _ray_box_intersect(ray_origins, ray_directions, ray_maxes,
            box_mins, box_maxes):

    div0_mask = torch.abs(ray_directions) < 1e-8
    inv_dir = 1 / (div0_mask * 1e-8 + ray_directions)
    
    t_mins = (box_mins - ray_origins) * inv_dir
    t_maxes = (box_maxes - ray_origins) * inv_dir

    tmin = torch.max(torch.minimum(t_mins, t_maxes), dim=1).values
    tmax = torch.min(torch.maximum(t_mins, t_maxes), dim=1).values
    intersected = (ray_maxes >= tmin) & (tmax > tmin) & (tmax > 0)

    return intersected, tmin, tmax


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
            


    





    