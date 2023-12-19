import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

from src import source_loader, cloud_gen, simple_depth_est, make_mesh


def main(transforms_path):

    # Load all scene data including camera parameters and all colour, depth, 
    # normal maps.
    scene_data = source_loader.load(transforms_path)

    #simple_depth_est.run(scene_data)

    clouds, pts, colours, normals = cloud_gen.cloud_gen(scene_data)

    make_mesh.run(clouds, pts, colours, normals)
    

if __name__ == '__main__':
    transforms_path = 'C:/Users/Andrew/colmap-projects/forest/transforms.json'
    main(transforms_path)