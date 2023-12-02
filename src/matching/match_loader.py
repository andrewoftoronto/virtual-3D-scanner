from typing import Dict
import numpy as np


class FrameFeatures:
    def __init__(self, id, frame, n_points):
        self.id = id
        self.frame = frame
        self.coords_2D = np.zeros(shape=[n_points, 2], dtype=float)
        self.point_3D_ids = np.zeros(shape=[n_points], dtype=np.int64)


class PointData:
    def __init__(self, n_points: int, id_to_frame_features: Dict[int, FrameFeatures]):
        self.id_to_frame_features: Dict[int, FrameFeatures] = id_to_frame_features
        self.id_to_point_index: Dict[int, int] = {}
        self.ids = np.zeros(shape=[n_points], dtype=np.int64)
        self.coords = np.zeros(shape=[n_points, 3], dtype=float)
        self.errs = np.zeros(shape=[n_points], dtype=float)

        # [[(frame, x, y), ...], ...]
        self.matches = []


class Point3D:
    def __init__(self, id, point_2Ds):
        self.id = id
        self.point_2Ds = point_2Ds


def load_matches(points_3D_file: str, points_2D_file: str) -> PointData:
    ''' Load matching feature data from the given files. '''

    # Load the 2D points file.
    id_to_frame_features = {}
    with open(points_2D_file) as f:
        
        # Skip header rows.
        lines = f.readlines()[4:]

        for i in range(0, len(lines), 2):
            image_info_line = lines[i].strip()
            image_info_tokens = image_info_line.strip().split(' ')
            id = int(image_info_tokens[0])
            frame_name = image_info_tokens[-1]

            point_info_line = lines[i + 1]
            point_info_tokens = point_info_line.split(' ')

            n_points = len(point_info_tokens) // 3
            frame_features = FrameFeatures(id, frame_name, n_points)
            id_to_frame_features[id] = frame_features

            for j in range(0, len(point_info_tokens), 3):
                x = float(point_info_tokens[j])
                y = float(point_info_tokens[j + 1])
                point3D_id = int(point_info_tokens[j + 2])
                
                frame_features.coords_2D[j // 3] = [x, y]
                frame_features.point_3D_ids[j // 3] = point3D_id

    # Load the 3D points file.
    with open(points_3D_file) as f:

        # Skip header rows.
        lines = f.readlines()[3:]

        point_data = PointData(len(lines), id_to_frame_features)
        for (point_index, line) in enumerate(lines):
            line = line.strip()
            tokens = line.split(' ')

            id = int(tokens[0])
            x = float(tokens[1])
            y = float(tokens[2])
            z = float(tokens[3])
            error = float(tokens[7])

            point_data.id_to_point_index[id] = point_index
            point_data.ids[point_index] = id
            point_data.coords[point_index] = [x, y, z]
            point_data.errs[point_index] = error

            # Process (image_id, point_2D) pairs.
            tokens = tokens[8:]
            point_2Ds = []
            for i in range(0, len(tokens), 2):
                image_id = int(tokens[i])
                point2D_idx = int(tokens[i + 1])

                frame_features: FrameFeatures = id_to_frame_features[image_id]
                frame = frame_features.frame
                x2D = frame_features.coords_2D[point2D_idx, 0]
                y2D = frame_features.coords_2D[point2D_idx, 1]
                point_2Ds.append((frame, x2D, y2D))

            point_data.matches.append(point_2Ds)

    return point_data

