import numpy as np
import os

class FeatureLoader():

    def __init__(self, feature_path):
        self.feature_path = feature_path

    def load(self, frame_id):
        filename = '{}.csv'.format(frame_id+1) # filename begins with 1.csv
        filepath = '{}neighbor_id/{}'.format(self.feature_path, filename)
        if not os.path.exists(filepath):
            return [], [], [], []
        with open(filepath) as f:
            lines = f.readlines()
            neighbor_ids = [int(l)-1 for l in lines]
        with open('{}viewIds_matches/{}'.format(self.feature_path, filename)) as f:
            lines = f.readlines()
            match_frame_ids = []
            for l in lines:
                tmp = []
                for x in l.split(',')[1:]:
                    if int(x) == -1:
                        break
                    tmp.append(neighbor_ids[int(x)-2])
                match_frame_ids.append(tmp)
        with open('{}points_matches/{}'.format(self.feature_path, filename)) as f:
            lines = f.readlines()
            lines = [[float(x) for x in l.split(',')] for l in lines]
            uv_points = [np.array(l) for l in lines]
            uv_points = [p[p!=-1].reshape(-1,2) for p in uv_points]
        with open('{}feature_matches_3d/{}'.format(self.feature_path, filename)) as f:
            lines = f.readlines()
            xyz_points = [[float(x) for x in l.split(',')] for l in lines]
        return neighbor_ids, match_frame_ids, uv_points, xyz_points
if __name__=='__main__':
    feature_loader = FeatureLoader('data/chess/features/')
    neighbor_ids, match_frame_ids, uv_points, xyz_points = feature_loader.load(0)