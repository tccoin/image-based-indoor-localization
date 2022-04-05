import argparse
import os
import tensorflow as tf
from siamese_network import posenet
from siamese_network.helper import readDataFromFile, parse_function
from ba.visual_isam2 import VisualISAM2
from search import search


class ImageBasedLocalization:
    def __init__(self, args):
        # set attr
        self.model_info_path = 'model/{}/training_data_info/{}_posenet_data.h5py'.format(
            args.model_name,
            args.model_name
        )
        self.map_data_file = 'data/tfrecord2/train.tfrecords'
        self.test_data_file = 'data/tfrecord2/test.tfrecords'
        self.map_feature_file = 'data/tfrecord2/test.tfrecords'
        self.test_feature_file = 'data/tfrecord2/test.tfrecords'

        # init model
        self.model, self.batch_size = self.load_model()

        # init isam2
        self.visam2 = VisualISAM2()

    def load_model(self):
        data_dict = readDataFromFile(self.model_info_path)
        model = posenet.create_siamese_keras()
        model.load_weights(data_dict['posenet_weight_file'])
        batch_size = data_dict['posenet_batch_size']
        return model, batch_size

    def load_tfrecord_dataset(self, dataset_path):
        dataset = tf.data.TFRecordDataset(
            dataset_path,
            compression_type='GZIP'
        )
        dataset = dataset.map(parse_function)
        poses = [x[1].numpy() for x in dataset]
        return dataset, poses
    
    def load_features(self, feature_path):
        pass
    
    def generate_map(self):
        dataset, poses = self.load_tfrecord_dataset(self.map_data_file)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(5)
        self.map_descriptors = self.model.predict(dataset)
        self.map_poses = poses
        # upload map to GPU for fast search
        search.init_data(self.map_descriptors)
        # todo: save map

    def load_map(self, map_path):
        pass

    def search_image(self, dataset):
        image_descriptor = self.model.predict(dataset)[0]
        return search.find_nearest(image_descriptor)

    def get_neighbors(self, i, k):
        # todo: return knn for i-th map frame, includeing poses, features
        pass

    def triangulate_landmarks(self, map_frame_id, current_features):
        neighbors = self.get_neighbors(map_frame_id, 10)
        # todo: triangulate each landmark in current_features
        pass

    def run_localization(self):
        self.generate_map()
        dataset, poses = self.load_tfrecord_dataset(self.test_data_file)
        features = self.load_features(self.test_feature_file)
        trajectory =  []
        for i in range(dataset):
            map_frame_id = self.search_image(dataset[i])
            triangulated_features = self.triangulate_landmarks(map_frame_id, features[i])
            current_pose = self.visam2.update(
                self.map_poses[map_frame_id], triangulated_features
            )
            trajectory.append(current_pose)
        return trajectory


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
    ibl.search_image()
