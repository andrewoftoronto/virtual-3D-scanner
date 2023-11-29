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
        self.transform_matrix = np.array(transform_matrix)


class TransformsData:
    def __init__(self, parameters: ProjectionParameters, frames: List[Frame]):
        self.parameters = parameters
        self.frames = frames


def read_transforms(file_path: str) -> TransformsData:
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
        frames.append(Frame(frame_file_path, transform_matrix))

    return TransformsData(params, frames)



