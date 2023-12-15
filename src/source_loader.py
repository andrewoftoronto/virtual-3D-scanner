from typing import List, Optional
import os
import numpy as np
import cv2
from PIL import Image
from .transforms import ProjectionParameters, Frame as TFrame, read_transforms
from .matching.match_loader import load_matches, PointData

class DelayLoadMap:
    def __init__(self, path):
        self.path = path
    def load(self):
        return load_image(self.path)

class Frame:
    ''' Info about one frame of source data, including camera transform and 
        the colour/normal/depth maps.'''
    
    def __init__(self, index, transform_frame: Optional[TFrame], colour_path: str, colour_map, depth_map, normal_map):
        self.index = index
        self.transform = transform_frame.transform_matrix if transform_frame is not None else None
        self.colour_path = colour_path
        self.colour_map = colour_map
        self.depth_map = depth_map
        self.normal_map = normal_map
        if "085" in colour_path:
            print("Loaded 85")

    def get_position(self):
        return self.transform[3,:3]

    def get_base_name(self):
        return os.path.split(self.colour_path)

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

    def __init__(self, transform_scene_data: ProjectionParameters, frames: List[Frame], extra_frames: List[Frame]):
        self.proj_params = transform_scene_data
        self.frames = frames
        self.extra_frames = extra_frames
        self.feature_data : PointData = None

    def lookup(self, frame_search_term: str):
        for frame in self.frames:
            if frame_search_term in frame.colour_path:
                return frame
        raise Exception(f"Unable to find frame matching query: {frame_search_term}") 

    def save_colours(self):
        ''' Save all colour images. '''
        for frame in self.frames:
            image = Image.fromarray(frame.get_colour())
            image.save(frame.colour_path)


def load(transforms_path: str, foggy: bool = False):
    ''' Load scene data from the given folder, identified by its transforms 
        file. '''
    colour_folder = "images" if not foggy else "foggy-images"

    # Get the path of the folder containing the transforms file.
    base_path = os.path.dirname(os.path.abspath(transforms_path))

    transforms = read_transforms(transforms_path, foggy)

    frames = []
    for (index, tframe) in enumerate(transforms.frames):
        image_file_path = os.path.join(base_path, tframe.file_path)
        colour = _load_map(image_file_path, colour_folder, base_path)
        depth = _load_map(image_file_path, 'depth', base_path)
        normal = _load_map(image_file_path, 'normal', base_path)
        frame = Frame(index, tframe, image_file_path, colour, depth, normal)
        frames.append(frame)

    transform_parameters = transforms.parameters
    del transforms

    # Sort frames by name.
    #frames.sort(key=lambda f: f.get_base_name())

    extra_images_path = os.path.join(base_path, "extra-images")
    extra_frame_names = []
    if os.path.exists(extra_images_path):
        extra_frame_names = os.listdir(extra_images_path)
    extra_frames = []
    for (index, name) in enumerate(extra_frame_names):
        image_file_path = os.path.join(extra_images_path, name)
        colour = _load_map(image_file_path, 'extra-images', base_path)
        depth = _load_map(image_file_path, 'extra-depth', base_path)
        normal = _load_map(image_file_path, 'extra-normal', base_path)
        frame = Frame(index, None, image_file_path, colour, depth, normal)
        extra_frames.append(frame)

    scene_data = RawSceneData(transform_parameters, frames, extra_frames)

    # Load 3D feature matches. We also correct each image to have the actual
    # Frame rather than just the name of the image.
    points_3D_path = os.path.join(base_path, "colmap_text/points3D.txt")
    points_2D_path = os.path.join(base_path, "colmap_text/images.txt")
    feature_data = load_matches(points_3D_path, points_2D_path)
    for (_id, frame_features) in feature_data.id_to_frame_features.items():
        frame_features.frame = scene_data.lookup(frame_features.frame)

    scene_data.feature_data = feature_data
    return scene_data


def _load_map(colour_path: str, type: str, base_path: str = None):
    valid_types = ['images', 'foggy-images', 'depth', 'normal', 
            'extra-images', 'extra-depth', 'extra-normal']

    if not os.path.isabs(colour_path):
        path = os.path.join(base_path, colour_path)
    else:
        path = colour_path

    if type not in valid_types:
        raise Exception(f"Invalid map type to load: {type}")
    
    path = os.path.abspath(path).replace('\\', '/')
    if type != 'images' and type != 'extra-images' and type != 'foggy-images':
        path = path.replace('extra-images/', f"{type}/").replace('foggy-images/', f"{type}/").replace('images/', f"{type}/")

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