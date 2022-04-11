import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import tensorflow as tf
from siamese_network import posenet
from siamese_network.helper import readDataFromFile, parse_function, angleDifference, distDifference
from ba.visual_isam2 import VisualISAM2
from search import search
from utils.feature_loader import FeatureLoader
import gtsam
import numpy as np

class ImageBasedLocalization:
    def __init__(self, args):
        # set attr
        self.model_info_path = 'model/{}/training_data_info/{}_posenet_data.h5py'.format(
            args.model_name,
            args.model_name
        )
        self.map_data_file = 'data/{}/tfrecord2/train.tfrecords'.format(args.model_name)
        self.test_data_file = 'data/{}/tfrecord2/test.tfrecords'.format(args.model_name)
        self.map_feature_file = 'data/{}/tfrecord2/test.tfrecords'.format(args.model_name)
        self.test_feature_file = 'data/{}/tfrecord2/test.tfrecords'.format(args.model_name)
        self.feature_path = 'data/{}/features/'.format(args.model_name)
        # init model
        self.model, self.batch_size = self.load_model()

        # init isam2
        self.visam2 = VisualISAM2()

    def load_model(self):
        data_dict = readDataFromFile(self.model_info_path)
        model = posenet.create_siamese_keras()
        model.load_weights(data_dict['posenet_weight_file'])
        model = tf.keras.Model(
            inputs=model.layers[2].input,
            outputs=model.layers[2].get_layer('cls3_fc_pose_xyz').output
        )
        batch_size = data_dict['posenet_batch_size']
        return model, batch_size

    def load_tfrecord_dataset(self, dataset_path):
        dataset = tf.data.TFRecordDataset(
            dataset_path,
            compression_type='GZIP'
        )
        dataset = dataset.map(parse_function)
        poses = [x[1].numpy() for x in dataset]
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(5)
        return dataset, poses
    
    def load_features(self, feature_path):
        pass
    
    def generate_map(self):
        dataset, poses = self.load_tfrecord_dataset(self.map_data_file)
        self.map_descriptors = self.model.predict(dataset)
        self.map_poses = poses
        # upload map to GPU for fast search
        search.init_data(self.map_descriptors)
        # todo: save map

    def load_map(self, map_path):
        pass

    def search_image(self, descriptor):
        return search.find_nearest(descriptor)

    def run_localization(self):
        self.generate_map()
        dataset, poses = self.load_tfrecord_dataset(self.test_data_file)
        descriptors = self.model.predict(dataset)
        feature_loader = FeatureLoader(self.feature_path)
        estimated_trajectory =  []
        num_gap = 1
        total = 50
        N = total//num_gap
        dist_diff = np.zeros(N)
        angle_diff = np.zeros(N)
        trajectory = []
        for i in range(0,total,num_gap):
            descriptor = descriptors[i]
            map_frame_id = self.search_image(descriptor)
            neighbor_ids, match_frame_ids, uv_points, xyz_points = feature_loader.load(i)
            if len(neighbor_ids)==0:
                dist_diff[i//num_gap], angle_diff[i//num_gap] = 0,0
                N -= 1
                continue
            pose_initial_guess = self.map_poses[map_frame_id]
            neighbor_poses = [self.map_poses[x] for x in neighbor_ids]
            result_poses = self.visam2.update(
                neighbor_ids, match_frame_ids, uv_points, xyz_points, pose_initial_guess, neighbor_poses, poses[i]
            )
            # estimated_pose = self.visam2.update_smart_factor(
            #     neighbor_ids, match_frame_ids, uv_points, pose_initial_guess, neighbor_poses
            # )
            dist_diff[i//num_gap], angle_diff[i//num_gap] = self.calc_error(poses[i], result_poses[1])
            print(i, 'dist diff:', dist_diff[i//num_gap],'angle diff:', angle_diff[i//num_gap])
            # self.visam2.plot(gtsam.Pose3(
            #     gtsam.Rot3.Quaternion(poses[i][6],*poses[i][3:6]),
            #     gtsam.Point3(poses[i][0:3])
            # ))
            trajectory.append(result_poses)
        print(np.median(dist_diff), np.median(angle_diff))
        return trajectory

    def calc_error(self, gt_pose, estimated_pose):
        quat = estimated_pose.rotation().quaternion()
        quat = np.hstack([quat[1:],quat[0]])
        dist_diff = distDifference(estimated_pose.translation(), gt_pose[0:3])
        angle_diff = angleDifference(quat, gt_pose[3:])
        return dist_diff, angle_diff
            

    def cleanup(self):
        search.cleanup()


def add_argument(parser):
    parser.add_argument('--model_name',
                        default='chess',
                        help='model name',
                        action='store'
                        )


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    print("flags argument: {}".format(args))

    ibl = ImageBasedLocalization(args)
    ibl.run_localization()
    ibl.cleanup()
