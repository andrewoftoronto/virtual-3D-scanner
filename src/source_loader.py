from typing import List
import os
import numpy as np
import cv2
from PIL import Image
from .transforms import TransformsData, Frame as TFrame, read_transforms


class DelayLoadMap:
    def __init__(self, path):
        self.path = path
    def load(self):
        return load_image(self.path)

class Frame:
    ''' Info about one frame of source data, including camera transform and 
        the colour/normal/depth maps.'''
    
    def __init__(self, transform_frame: TFrame, colour_path: str, colour_map, depth_map, normal_map):
        self.transform = transform_frame.transform_matrix
        self.colour_path = colour_path
        self.colour_map = colour_map
        self.depth_map = depth_map
        self.normal_map = normal_map
        if "085" in colour_path:
            print("Loaded 85")

    def get_colour(self):
        if isinstance(self.colour_map, DelayLoadMap):
            self.colour_map = self.colour_map.load()
        return self.colour_map
    def get_normal(self):
        if isinstance(self.normal_map, DelayLoadMap):
            self.normal_map = self.normal_map.load()
        return self.normal_map
    def get_depth(self):
        if isinstance(self.depth_map, DelayLoadMap):
            self.depth_map = self.depth_map.load()
        return self.depth_map


class RawSceneData:
    ''' Collection of all source data used to describe a scene. '''

    def __init__(self, transform_scene_data: TransformsData, frames: List[Frame]):
        self.proj_params = transform_scene_data.parameters
        self.frames = frames


def load(transforms_path: str):
    ''' Load transforms from the given path. '''

    # Get the path of the folder containing the transforms file.
    base_path = os.path.dirname(os.path.abspath(transforms_path))

    transforms = read_transforms(transforms_path)

    frames = []
    for tframe in transforms.frames:
        colour = _load_map(tframe.file_path, 'colour', base_path)
        depth = _load_map(tframe.file_path, 'depth', base_path)
        normal = _load_map(tframe.file_path, 'normal', base_path)
        frame = Frame(tframe, tframe.file_path, colour, depth, normal)
        frames.append(frame)

    return RawSceneData(transforms, frames)


def _load_map(colour_path: str, type: str, base_path: str = None):
    valid_types = ['colour', 'depth', 'normal']

    if not os.path.isabs(colour_path):
        path = os.path.join(base_path, colour_path)
    else:
        path = colour_path

    if type not in valid_types:
        raise Exception(f"Invalid map type to load: {type}")
    
    if type != 'colour':
        path = path.replace('images/', f"{type}/")

    path = os.path.abspath(path)
    image = _load_image_without_extension(path)
    return image


def load_image(image_path):
    print(f"Loading: {image_path}")
    if image_path.endswith('.exr'):
        image = cv2.imread(image_path,  cv2.IMREAD_UNCHANGED)
        if image.shape[-1] >= 3:
            image[:, :, 0], image[:, :, 2] = image[:, :, 2].copy(), image[:, :, 0].copy()
    else:
        image = np.array(Image.open(image_path))
    return image


def _load_image_without_extension(file_path: str):
    # Get the directory and basename of the file
    directory, base_name = os.path.split(file_path)
    base_name_no_extension, _ = os.path.splitext(base_name)

    # Search for files with the same basename and different extensions
    matching_files = [
        os.path.join(directory, file) for file in os.listdir(directory)
        if file.startswith(base_name_no_extension)
    ]

    if not matching_files:
        raise FileNotFoundError(f"No matching files found for {file_path}")

    # Load the first matching image (you may want to add more logic based on your requirements)
    image_path = matching_files[0]
    return DelayLoadMap(image_path)