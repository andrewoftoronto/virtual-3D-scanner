''' Encode the scene with AI operators. '''

import numpy as np
import torch
from .source_loader import RawSceneData
from .projection import unproject


N_BITS_PER_BLOCK_DIM = 20


def run(scene_data: RawSceneData):
    ''' Run the ai-encoder on the given scene. '''
    
    filled_block_indices, minimum, maximum = _identify_filled_blocks(scene_data=scene_data)
    _construct_octree(
        filled_block_indices_np=filled_block_indices.cpu().numpy(), 
        minimum_np=minimum.cpu().numpy(),  
        maximum_np=maximum.cpu().numpy())


def _identify_filled_blocks(scene_data: RawSceneData):
    ''' Identifies each block in 3D space that contains at least one surface
        sample. '''

    filled_block_indices = torch.zeros(size=[0], dtype=torch.int64).cuda()
    minimum = None
    maximum = None
    for frame in scene_data.frames:

        # Find blocks that each world point would belong in.
        B = N_BITS_PER_BLOCK_DIM
        world_points = unproject(scene_data, frame, "cuda")
        block_coords = torch.floor(world_points).to(torch.int64)
        this_min, _ = torch.min(block_coords, dim=0)
        this_max, _ = torch.max(block_coords, dim=0)
        if minimum is None:
            minimum = this_min
            maximum = this_max
        else:
            minimum = torch.minimum(minimum, this_min)
            maximum = torch.maximum(minimum, this_max)

        # Express block coordinates as a single spatial index value. 
        # We add (1 << (B - 1)) to prevent negatives.
        block_indices = block_coords[:, 0] + (block_coords[:, 1] << B) + (block_coords[:, 2] << (2 * B))

        # Add to our knowledge set of filled blocks.
        uniques = torch.unique(block_indices)
        filled_block_indices = torch.concat((filled_block_indices, uniques))
        filled_block_indices = torch.unique(filled_block_indices)
    
    return filled_block_indices, minimum, maximum


def _construct_octree(filled_block_indices_np, minimum_np, maximum_np):
    ''' Construct an octree using the indices of the filled blocks. '''

    # Create tree with root. n_levels excludes leaves.
    greatest_range = (maximum_np - minimum_np).max()
    n_levels = np.ceil(np.log2(greatest_range)).astype(np.int64)
    tree = -np.ones(shape=[1, 8], dtype=np.int64)

    # Decode block indices.
    B = N_BITS_PER_BLOCK_DIM
    x_mask = (1 << B) - 1
    y_mask = ((1 << B) - 1) << B
    z_mask = ((1 << B) - 1) << (2 * B)
    filled_block_indices_np = filled_block_indices_np.reshape([-1, 1])
    x = (filled_block_indices_np & x_mask) - minimum_np[0]
    y = ((filled_block_indices_np & y_mask) >> B) - minimum_np[1]
    z = ((filled_block_indices_np & z_mask) >> (2 * B)) - minimum_np[2]
    coords = np.concatenate((x, y, z), axis=1)

    # Add each block to the tree one at a time.
    n_blocks = len(filled_block_indices_np)
    for i in range(0, n_blocks):

        # Note level here ranges in [1, n_levels]. Since leaves are excluded,
        # this corresponds to the actual logarithmic physical size of the level.
        relative_coords = coords[i]
        node_index = 0
        for level in range(n_levels, 0, -1):

            # Find which of the 8 children the block is in.
            mid = (1 << (level - 1))
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

            node_index = child_index









    