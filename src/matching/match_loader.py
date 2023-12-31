from typing import Dict
import numpy as np


class FrameFeatures:
    def __init__(self, id, frame, coords_2D, point_3D_ids):
        self.id = id
        self.frame = frame
        self.coords_2D = np.array(coords_2D, dtype=float)
        self.point_3D_ids = np.array(point_3D_ids, dtype=np.int64)


class PointData:
    def __init__(self, n_points: int, id_to_frame_features: Dict[int, FrameFeatures]):
        self.id_to_frame_features: Dict[int, FrameFeatures] = id_to_frame_features
        self.id_to_point_index: Dict[int, int] = {}
        self.ids = np.zeros(shape=[n_points], dtype=np.int64)
        self.coords = np.zeros(shape=[n_points, 3], dtype=float)
        self.errs = np.zeros(shape=[n_points], dtype=float)

        # [[(frame_features_index, x, y), ...], ...]
        self.matches = []

    def n_frames(self) -> int:
        ''' Gets the total number of frames in this dataset. '''
        return len(self.id_to_frame_features)

    def n_features(self) -> int:
        ''' Gets the total number of features in this dataset. '''
        return len(self.id_to_point_index)
    
    def n_frame_features(self) -> int:
        ''' Gets the total number of 2D points in this dataset. '''

        count = 0
        for frame_features in self.id_to_frame_features.values():
            count += len(frame_features.coords_2D)
        return count
    
    def filter_by_err(self, n_best_per_frame):
        ''' Reject features that aren't among n_best_per_frame in at least one
            frame. '''
        
        # Each frame identifies its top n_best_per_frame features.
        frame_bests = [[] for i in range(0, self.n_frames())]
        for frame_features in self.id_to_frame_features.values():
            frame_index = frame_features.frame.index

            frame_feature_indices = []
            frame_feature_errs = []
            for point_3D_id in frame_features.point_3D_ids:
                index_3D = self.id_to_point_index[point_3D_id]
                err = self.errs[index_3D]
                frame_feature_indices.append(index_3D)
                frame_feature_errs.append(err)

            frame_feature_indices = np.array(frame_feature_indices)
            frame_feature_errs = np.array(frame_feature_errs)

            # Apply err sorting to frame_feature_indices.
            sorting = np.argsort(frame_feature_errs)
            frame_feature_indices = frame_feature_indices[sorting]
            del frame_feature_errs

            frame_bests[frame_index] = frame_feature_indices[:n_best_per_frame]

        # Aggregate these ratings together.
        indices_to_keep = set(index for frame_best_list in frame_bests for index in frame_best_list)
        self.filter_by_index(list(indices_to_keep))

    def _rebuild_id_to_point_index(self):
        self.id_to_point_index = {id: index for (index, id) in enumerate(self.ids)}

    def filter_by_index(self, indices_to_keep):
        ''' Filter 3D points by their index. The 2D and 3D data for any point
            being filtered out will be deleted. '''
        
        # 3D.
        indices_to_keep = np.array(indices_to_keep)
        self.coords = self.coords[indices_to_keep]
        self.errs = self.errs[indices_to_keep]
        self.ids = self.ids[indices_to_keep]
        self._rebuild_id_to_point_index()
        self.matches = [value for index, value in enumerate(self.matches) if index in indices_to_keep]

        # 2D.
        for frame_features in self.id_to_frame_features.values():
            mask = np.isin(frame_features.point_3D_ids, self.ids)
            frame_features.point_3D_ids = frame_features.point_3D_ids[mask]
            frame_features.coords_2D = frame_features.coords_2D[mask]







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

            points_2D = []
            points_3D_id = []
            for j in range(0, len(point_info_tokens), 3):
                x = float(point_info_tokens[j])
                y = float(point_info_tokens[j + 1])
                point3D_id = int(point_info_tokens[j + 2])

                points_2D.append((x, y))
                points_3D_id.append(point3D_id)

            frame_features = FrameFeatures(id, frame_name, points_2D, points_3D_id)
            id_to_frame_features[id] = frame_features

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

                frame_features = id_to_frame_features[image_id]
                x2D = frame_features.coords_2D[point2D_idx, 0]
                y2D = frame_features.coords_2D[point2D_idx, 1]
                point_2Ds.append((frame_features, x2D, y2D))

            point_data.matches.append(point_2Ds)

    # Eliminate 2D points with no corresponding 3D one.
    for frame_features in point_data.id_to_frame_features.values():
        filter_mask = frame_features.point_3D_ids != -1
        frame_features.point_3D_ids = frame_features.point_3D_ids[filter_mask]
        frame_features.coords_2D = frame_features.coords_2D[filter_mask]

    return point_data

