import os
import json
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

from src import source_loader, ai_encode, simple_depth_est
from src import cloud_gen, make_mesh


def main(transforms_path):
    scene_root = os.path.split(transforms_path)[0]

    # Load all scene data including camera parameters and all colour, depth, 
    # normal maps.
    scene_data = source_loader.load(transforms_path)

    # Convert to SDF AI scene encoding.
    camera_file_path = os.path.join(scene_root, "camera.json")
    if os.path.exists(camera_file_path):
        with open(camera_file_path) as f:
            camera_params = json.load(f)
            scene_data.proj_params.znear = camera_params["znear"]
            scene_data.proj_params.zfar = camera_params["zfar"]
    else:
        simple_depth_est.run(scene_data)
        with open(camera_file_path, 'w') as json_file:
            data = {
                "znear": scene_data.proj_params.znear,
                "zfar": scene_data.proj_params.zfar
            }
            json.dump(data, json_file)

    ai_encode.run(scene_data)

    clouds, pts, colours, normals = cloud_gen.cloud_gen(scene_data)
    make_mesh.run(clouds, pts, colours, normals)
    

if __name__ == '__main__':
    transforms_path = 'C:/Users/Andrew/colmap-projects/forest2/transforms.json'
    main(transforms_path)