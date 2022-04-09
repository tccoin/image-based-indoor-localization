import helper
import posenet
import tensorflow as tf
import random
import os.path
from datetime import datetime
from random import uniform
import shutil
import argparse

from helper import parse_function, initialise_callbacks, custom_generator, customPredictGenerator_tf2, PreprocessingData, readDataFromFile
random.seed(datetime.now())
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

class Test():
    def __init__(self, args):
        
        # data paths
        self.input_data_dir = args.input_data_dir
        self.output_data_dir = args.output_data_dir
        self.training_data = args.training_data
        self.validation_data = args.validation_data
        self.testing_data = args.testing_data

        # training, validation and testing data
        self.input_train_tfrecord = [self.input_data_dir + file_name + '.tfrecords' for file_name in self.training_data]
        self.testing_data = [self.input_data_dir + file_name + '.tfrecords' for file_name in self.testing_data]
        print('testing data: {}'.format(self.testing_data))
        self.testing_name = args.testing_name
        
        self.should_plot_cdf = True
        self.should_plot = True
        self.should_save_output_data = True
        
        self.training_info_file = self.output_data_dir  + 'training_data_info/{}_posenet_data.h5py'.format(self.testing_name)
        self.weights_info_file = self.output_data_dir + 'weights/weights_info_posenet.csv'
        self.save_cdf_position_file = self.output_data_dir + 'cdf/{}_position_cdf.png'.format(self.testing_name)
        self.save_cdf_orientation_file = self.output_data_dir + 'cdf/{}_orientation_cdf.png'.format(self.testing_name)
        self.save_trajectory_file = self.output_data_dir + 'cdf/{}_trajectory.png'.format(self.testing_name)
        self.save_output_data_file = self.output_data_dir + 'cdf/{}_output.h5py'.format(self.testing_name)
        self.save_op_per_seq = self.output_data_dir + 'cdf/{}_each_sequence.csv'.format(self.testing_name)
        
        self.diff_location = args.diff_location
        # self.num_img_siamese = args.num_img_siamese
        self.num_candidate = args.num_candidate
        self.testing_2d_flag = False

        self.threshold_angle = args.threshold_angle
        self.threshold_dist = args.threshold_dist

    def test_posenet(self):
        
        # Get data from training run
        data_dict = readDataFromFile(self.training_info_file)
        
        # Test model
        model = posenet.create_posenet_keras()
        
        if self.diff_location:
            batch_size = 64
            model_file_path = self.output_data_dir  + 'weights/{}.h5'.format(self.testing_name)
            
            # Reading data file
            self.input_test_tfrecord = []
            
            dataset = tf.data.TFRecordDataset(self.testing_data,compression_type='GZIP')
            
            print('model file loading: {}'.format(self.testing_name))
            model.load_weights(model_file_path)
        else:
            try:
                print('model file loading: {}'.format(data_dict['posenet_checkpoint_weight_file']))
                model.load_weights(data_dict['posenet_checkpoint_weight_file'])
            except:
                model.load_weights(data_dict['posenet_weight_file'])
        
            # Reading data file
            dataset = tf.data.TFRecordDataset(data_dict['posenet_dataset_test'],compression_type='GZIP')
            batch_size=data_dict['posenet_batch_size']
            valid_size=data_dict['posenet_valid_size']
        
        ### Dataset API
        dataset_test = dataset.map(parse_function)
        dataset_test = dataset_test.batch(batch_size)
        dataset_test = dataset_test.batch(valid_size)
        dataset_test = dataset_test.prefetch(5)

        median_result = customPredictGenerator_tf2(
            model,
            dataset_test,
            batch_size,
            training_name=self.testing_name,
            should_plot=self.should_plot,
            should_plot_cdf=self.should_plot_cdf,
            should_save_output_data=self.should_save_output_data,
            save_cdf_position_file=self.save_cdf_position_file,
            save_cdf_orientation_file=self.save_cdf_orientation_file,
            save_trajectory_file=self.save_trajectory_file,
            save_output_data_file=self.save_output_data_file,
            flag2D = self.testing_2d_flag
        )

        shutil.copyfile(self.weights_info_file, self.weights_info_file + '.tmp')

        with open(self.weights_info_file, 'r') as f:
            with open(self.weights_info_file + '.tmp', 'w') as g:
                for line in f:
                    if (self.testing_name in line) and ('Median error' not in line):
                        g.write('{}. {}\n'.format(line.rstrip('\n'), median_result))
                    else:
                        g.write(line)

        shutil.copyfile(self.weights_info_file + '.tmp',self.weights_info_file)
    
    def test_image_search(self):
        # Get data from training run
        data_dict = readDataFromFile(self.training_info_file)

        # Test model
        model =  posenet.create_siamese_keras()
        
        try:
            print('model file loading: {}'.format(data_dict['posenet_weight_file']))
            model.load_weights(data_dict['posenet_weight_file'])
        except:
            model.load_weights(data_dict['posenet_checkpoint_weight_file'])


        pose_train = []
        dataset_tr = tf.data.TFRecordDataset(self.input_train_tfrecord, compression_type='GZIP')
        dataset_train = dataset_tr.map(parse_function)
        for features in dataset_train:
            pose_train += [features[1].numpy()]

        based_model = tf.keras.Model(
            inputs=model.layers[2].input,
            outputs=model.layers[2].get_layer('cls3_fc_pose_xyz').output
        )

        aggregated_results = []
        batch_size=data_dict['posenet_batch_size']

        #   Get training data descriptions
        dataset_train = dataset_train.batch(batch_size)
        dataset_train = dataset_train.prefetch(5)
        train_results = based_model.predict(dataset_train)
        for dataset_name in ['test']:
            temp = []
            input_data_file = self.input_data_dir  + dataset_name + '.tfrecords'
            print('########## testing file ########### : {}'.format(input_data_file))
            dataset = tf.data.TFRecordDataset(input_data_file,compression_type='GZIP')

            pose_test = []
            dataset_test = dataset.map(parse_function)
            for features in dataset_test:
                pose_test += [features[1].numpy()]
            
            median_result = helper.customPredictSiamese_tf2(
                based_model,
                dataset_test,
                train_results,
                batch_size,pose_train,pose_test,
                self.threshold_angle,
                self.threshold_dist,
                self.num_candidate,
                save_output_data_file=self.save_output_data_file
            )

            temp.append(input_data_file)
            temp.append(median_result)
            aggregated_results.append(temp)
        ## write training name to csv file where results can be stored in future
        with open(self.save_op_per_seq, 'w') as f:
            for value in aggregated_results:
                f.write('%s\n'%value)

def add_argument(parser):

    # testing
    dataset_name = 'chess' # ethl1_local chess
    parser.add_argument('--testing_name', default=dataset_name, help='name of training', action='store')
    parser.add_argument('--test_posenet', default=False, help='test posenet', action='store')
    parser.add_argument('--test_image_search', default=True, help='test image search', action='store')

    #IO file locations Modify here dataset_name
    parser.add_argument('--output_data_dir', default='model/{}/'.format(dataset_name), help='location of outputs', action='store')
    parser.add_argument('--input_data_dir', default='data/{}/tfrecord2/'.format(dataset_name), help='location of training tfrecords file', action='store')
    parser.add_argument('--training_data', default=['train'], help=' list fo training tfrecord file', action='store')
    parser.add_argument('--validation_data', default=['test'], help='location of validation h5py file', action='store')
    parser.add_argument('--testing_data', default=['test'], help='location of testing h5py file', action='store')
    parser.add_argument('--pretrained_model', default='model/posenet.npy', help='pretrained googlenet model on places dataset', action='store')
    parser.add_argument('--diff_location', default=False, help='test model with location specified and not from the location in training_data_info.h5py', action='store')

    # Siamese
    parser.add_argument('--num_candidate', default=5, help='Number of candidate zones from the Siamese comparison',
                        action='store')
    parser.add_argument('--threshold_angle', default=15, help='Threshold of angle to segment image pairs',
                        action='store')
    parser.add_argument('--threshold_dist', default=0.3, help='Threshold of angle to segment image pairs',
                        action='store')


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    print("flags argument: {}".format(args))

    test = Test(args)
    if args.test_posenet:
        test.test_posenet()	
    if args.test_image_search:
        test.test_image_search()
    


