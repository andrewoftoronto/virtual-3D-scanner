from typing import List
import json
import numpy as np


class ProjectionParameters:
    def __init__(self, fl_x: float, fl_y: float, cx: int, cy: int, w: int, h: int):
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h


class Frame:
    def __init__(self, file_path: str, transform_matrix: str):
        self.file_path = file_path

        # The given matrix is useful for the typical opengl multiplication 
        # order, but this project uses the opposite convention.
        self.transform_matrix = transform_matrix


class TransformsData:
    def __init__(self, parameters: ProjectionParameters, frames: List[Frame]):
        self.parameters = parameters
        self.frames = frames


def read_transforms(file_path: str, foggy: bool = False) -> TransformsData:
    ''' Read the transforms file. '''

    # Read the JSON file into a Python dictionary
    with open(file_path, 'r') as file:
        data = json.load(file)

    params = ProjectionParameters(
            data['camera_angle_x'], data['camera_angle_y'], 
            data['cx'], data['cy'], data['w'], data['h'])
    
    dict_frames = data['frames']
    frames = []
    for frame in dict_frames:
        frame_file_path = frame['file_path']
        transform_matrix = frame["transform_matrix"]

        r = np.array([
            [1, 0, 0, 0],
            [0, -1.0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

        transform_matrix = np.array(transform_matrix).T
        transform_matrix = np.matmul(r, transform_matrix)

        if foggy:
            frame_file_path = frame_file_path.replace('images', 'foggy-images')

        frames.append(Frame(frame_file_path, transform_matrix))

    return TransformsData(params, frames)



