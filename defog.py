import os
import shutil
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from src import source_loader
from src.defogger import infer_fog_params, defog


''' Learns the fog parameters for a scene and then defogs it. '''

def main(transforms_path):

    # If a foggy folder does not exist, make one and move everything into it.
    foggy_path = os.path.join(os.path.split(transforms_path)[:-1][0], "foggy-images")
    if not os.path.exists(foggy_path):
        images_path = os.path.join(os.path.split(transforms_path)[0], "images")
        shutil.move(images_path, foggy_path)
        os.makedirs(images_path)

    scene_data = source_loader.load(transforms_path, foggy=True)
    infer_fog_params(scene_data)
    defog(scene_data)

    # Rename all frame colour images to not specify foggy folder anymore.
    for frame in scene_data.frames:
        frame.colour_path = frame.colour_path.replace('foggy-images', 'images')

    scene_data.save_colours()
    pass

if __name__ == '__main__':
    transforms_path = 'C:/Users/Andrew/colmap-projects/valheim/transforms.json'
    main(transforms_path=transforms_path)