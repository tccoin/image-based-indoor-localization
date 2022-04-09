import posenet
import tensorflow as tf
import numpy as np
from tensorflow import keras
import random
import os.path
from pathlib import Path
from datetime import datetime
import math
from random import uniform
import shutil
import argparse
import time
from scipy.spatial.transform import Rotation as R
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import gc
from helper import parse_function, initialise_callbacks, custom_generator, customPredictGenerator_tf2, PreprocessingData, readDataFromFile

random.seed(datetime.now())
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

class Train():

    def __init__(self, args):
        
        # data paths
        self.input_data_dir = args.input_data_dir
        self.output_data_dir = args.output_data_dir
        self.training_data = args.training_data
        self.validation_data = args.validation_data
        self.testing_data = args.testing_data
        self.pretrained_model_path = args.pretrained_model
        
        # Hyper paramters 
        self.batch_size = args.batch_size
        self.valid_size = args.valid_size
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.beta = args.beta
        self.decay = uniform(-5, -6) # not used as of now
        
        # training name 
        self.training_name        = args.training_name

        # gpu machine to use
        self.GPU_FLAG = args.GPU_FLAG
        
        # multiple GPU flag
        self.MULTIPLE_GPU_GLAG = args.MULTIPLE_GPU_GLAG
        
        #tensorboard and model checkpoint saving paths
        self.tensorboard_event_logs = args.tensorboard_event_logs
        self.csv_logs = args.csv_logs
        self.checkpoint_weights = args.checkpoint_weights
        self.weights = args.weights #location where final weights are stored
        self.training_data_info = args.training_data_info
        
        #loading data from different location num_img_siamese
        self.diff_location = args.diff_location
        # self.num_img_siamese = args.num_img_siamese
        self.date_time = None
        
        #function calls on initialization
        self.setupIOFiles()
        self.setTrainingParams()
            
    def contrastive_loss(self,y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        # y_pred = tf.cast(y_pred, dtype=tf.int64)
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    def accuracy(self,y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))	

    def angleDifference(self, q1, q2):

        angle = 0
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        r1 = r1.as_matrix()
        r2 = r2.as_matrix()
        if abs(np.trace(np.dot(r1.transpose(),r2))-3) < 0.0000000001:
            angle = 0
        else:
            angle = math.acos((np.trace(np.dot(r1.transpose(),r2))-1)/2) * 180 / np.pi
        return angle
    def distDifference(self, p1, p2):
        tmp = p1 - p2
        return np.linalg.norm(tmp)
    def getImageSimilarity(self, pose1, pose2, threshold_anlge, threshold_dist):

        distDif = self.distDifference(pose1[0:3], pose2[0:3])
        angleDif  = self.angleDifference(pose1[3:], pose2[3:])
        if (angleDif < threshold_anlge) and (distDif < threshold_dist):
            return True
        else:
            return False

    def train_pairs_generator(self,x_train,pose,threshold_angle,threshold_dist, num_data,batchsize): #num_size: training data size
        while 1:
            pairs_1 = []
            pairs_2 = []
            labels = []

            k = 0
            while k<batchsize:
                i = random.randrange(0, num_data-1)

                while True:
                    delta = random.randrange(-30, 30)
                    while (i + delta < 0 or i + delta >= num_data):
                        delta = random.randrange(-num_data, num_data)

                    ind = i + delta
                    bool_similar = self.getImageSimilarity(pose[i], pose[ind], threshold_angle, threshold_dist)
                    if bool_similar==True:
                        pairs_1 += [x_train[i]]
                        pairs_2 += [x_train[ind]]
                        labels += [1.0]
                        break

                while True:
                    ind = random.randrange(0, num_data - 1)
                    bool_similar = self.getImageSimilarity(pose[i], pose[ind], threshold_angle, threshold_dist)
                    if bool_similar==False:
                        pairs_1 += [x_train[i]]
                        pairs_2 += [x_train[ind]]
                        labels += [0.0]
                        break

                k = k+1
            pairs_1 = np.array(pairs_1)
            pairs_2 = np.array(pairs_2)
            labels = np.array(labels)
            # labels  = np.array(labels,dtype=np.float)
            yield [pairs_1,pairs_2], labels
    def valid_pairs_generator(self,x_train,pose,threshold_angle,threshold_dist, num_data,batchsize): #num_size: training data size
        pairs_1 = []
        pairs_2 = []
        labels = []

        k = 0
        while k < batchsize:
            i = random.randrange(0, num_data - 1)

            while True:
                delta = random.randrange(-30, 30)
                while (i + delta < 0 or i + delta >= num_data):
                    delta = random.randrange(-num_data, num_data)

                ind = i + delta
                bool_similar = self.getImageSimilarity(pose[i], pose[ind], threshold_angle, threshold_dist)
                if bool_similar == True:
                    pairs_1 += [x_train[i]]
                    pairs_2 += [x_train[ind]]
                    labels += [1.0]
                    break

            while True:
                ind = random.randrange(0, num_data - 1)
                bool_similar = self.getImageSimilarity(pose[i], pose[ind], threshold_angle, threshold_dist)
                if bool_similar == False:
                    pairs_1 += [x_train[i]]
                    pairs_2 += [x_train[ind]]
                    labels += [0.0]
                    break

            k = k + 1
        pairs_1 = np.array(pairs_1)
        pairs_2 = np.array(pairs_2)
        labels = np.array(labels)
        return [pairs_1,pairs_2], labels
    
    def setupIOFiles(self):
        self.date_time = datetime.now().strftime('%Y_%m_%d_%H_%M')
        self.input_train_tfrecord = []
        self.input_validation_tfrecord = []
        self.input_test_tfrecord = []
        for file_name in self.training_data:
            self.input_train_tfrecord.append(self.input_data_dir + str(file_name) + '.tfrecords')
        for file_name in self.validation_data:
            self.input_validation_tfrecord.append(self.input_data_dir + str(file_name) + '.tfrecords')
        for file_name in self.testing_data:
            self.input_test_tfrecord.append(self.input_data_dir + str(file_name) + '.tfrecords')
        
        print('training data list: {}'.format(self.input_train_tfrecord))
        print('validation data list: {}'.format(self.input_validation_tfrecord))
        print('testing data list: {}'.format(self.input_test_tfrecord))
        
        
    def setTrainingParams(self):
        self.training_info = '{} (posenet): Learning rate = {}. Batch size = {}. Beta = {}' \
            .format(self.training_name, self.lr, self.batch_size,self.valid_size, self.beta)

        self.tensor_log_file = self.output_data_dir + self.tensorboard_event_logs + '{}_posenet_tb_logs'.format(self.training_name)
        self.csv_log_file = self.output_data_dir + self.csv_logs + 'posenet_csv_{}.log'.format(self.training_name)
        self.weights_info_file = self.output_data_dir + self.weights + 'weights_info_posenet.csv'
        self.checkpoint_weights_file = self.output_data_dir +  self.checkpoint_weights + '{}_posenet.h5'.format(self.training_name)
        self.weights_out_file = self.output_data_dir + self.weights + 'posenet_weights_{}.h5'.format(self.training_name)
        self.output_preprocess_file = self.output_data_dir  + self.training_data_info + '{}_posenet_data.h5py'.format(self.training_name)
        self.model_json_file = self.output_data_dir + self.weights + 'model_{}.json'.format(self.training_name)
        self.threshold_angle = args.threshold_angle
        self.threshold_dist = args.threshold_dist
        self.csv_log_path = Path(self.csv_log_file)
        weights_out_path = Path(self.weights_out_file)


        assert not os.path.isdir(self.tensor_log_file), \
            "Log file already exist. Ensure" \
            " that the name you have entered" \
            " is unique. Names set in " \
            " TensorBoard callback."

        assert not self.csv_log_path.is_file(), \
            "Log file already exist. Ensure" \
            " that the name you have entered" \
            " is unique. Names set in " \
            " CSVLogger callback."

        if os.path.exists(self.weights_out_file):
            os.remove(self.weights_out_file)
        assert not weights_out_path.is_file(), \
            "Weight file already exists. Ensure" \
            " that the name you have entered" \
            " is unique. Name set in " \
            " model.save_weights()."
        
    def train(self):

        # Reading data file
        dataset_tr = tf.data.TFRecordDataset(self.input_train_tfrecord, compression_type='GZIP')
        dataset_val = tf.data.TFRecordDataset(self.input_validation_tfrecord, compression_type='GZIP')
    
        # init
        model = posenet.create_siamese_keras(self.pretrained_model_path, True)  # GoogLeNet (Trained on Places)
        adam = keras.optimizers.Adam(lr=self.lr, clipvalue=1.5)  # not sure if clipvalue is even considered
        model.compile(loss=self.contrastive_loss, optimizer=adam, metrics=[self.accuracy])

        model.summary()
        # save model graph 
        model_json = model.to_json()
        if os.path.exists(os.path.dirname(self.model_json_file)):
            shutil.rmtree(os.path.dirname(self.model_json_file))
            os.makedirs(os.path.dirname(self.model_json_file))
        with open(self.model_json_file, 'w') as f:
                f.write(model_json)
        f.close()
        # Setup callbacks
        callback_list = initialise_callbacks(self.checkpoint_weights_file, self.tensor_log_file, self.csv_log_file, self.batch_size, self.lr, self.num_epochs)
        
        ### Dataset API
        dataset_train = dataset_tr.map(parse_function)
        dataset_validation = dataset_val.map(parse_function)
            
        
        #calculate image mean
        preprocessing_data = PreprocessingData()
        num_trainimg = 0
        xtrain = []
        pose   = [] 
        for features in dataset_train:
            num_trainimg = num_trainimg + 1
            xtrain += [features[0].numpy()]
            pose   += [features[1].numpy()]
        te_pairs, te_y = self.valid_pairs_generator(
            xtrain,
            pose,
            self.threshold_angle,
            self.threshold_dist,
            num_trainimg,
            self.valid_size
        )

        # create training info dict which will be saved for either retraining or testing
        data_dict = {'posenet_learning_rate': self.lr,
                        'posenet_batch_size': self.batch_size,
                        'posenet_valid_size': self.valid_size,
                        'posenet_decay': self.decay,
                        'posenet_epochs': self.num_epochs,
                        'posenet_beta': self.beta,
                        'posenet_dataset_train': self.input_train_tfrecord,
                        'posenet_dataset_validation': self.input_validation_tfrecord,
                        'posenet_dataset_test': self.input_test_tfrecord,
                        'posenet_checkpoint_weight_file': self.checkpoint_weights_file,
                        'posenet_weight_file': self.weights_out_file}

        print('data dict: {}'.format(data_dict))
        preprocessing_data.writeDataToFile(self.output_preprocess_file, data_dict)
        print("#############Training details###############:\n Training name = {} and hyper parameters are: batch_size: {},valid_size: {}, lr: {}, beta: {} ".format(self.training_name, self.batch_size,self.valid_size, self.lr, self.beta))

        ## write training name to csv file where results can be stored in future
        with open(self.weights_info_file, 'a') as f:
            f.write(self.training_info)
            f.write('\n')
        f.close()

        # Fits the model. steps_per_epoch is the total number of batches to
        # yield from the generator before declaring one epoch finished and
        # starting the next. This is important since batches will be yielded
        # infinitely.
        t_start = time.time()
        print(t_start)
        version = tf.__version__
        print('version: {} and type: {}: {}'.format(version, type(version), int(version.split('.')[0])))
        epoch_per_each_training = 2
        cycles = math.floor(self.num_epochs/ epoch_per_each_training)
        previous_loss = 99999999999999999999999

        if (int(version.split('.')[0]) == 1 ):
            print('!!  Please use TENSORFLOW 2.0 !!')
        else:
            print('TENSORFLOW 2.0.0 SELECTED')

        model.fit(self.train_pairs_generator(xtrain,pose,self.threshold_angle,self.threshold_dist, num_trainimg,self.batch_size),
                                epochs=self.num_epochs,
                                steps_per_epoch=10,
                                validation_data=(te_pairs, te_y),
                                verbose=1,
                                callbacks=[ClearMemory()])

        model.save_weights(self.weights_out_file, overwrite=True)
    
        print("Training name = {} and hyper parameters are: batch_size: {}, lr: {}, beta: {} ".format(self.training_name, self.batch_size, self.lr, self.beta))

        t_end = time.time()
        print(t_end)
        print('time taken: {}'.format(t_end-t_start))

class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        K.clear_session()

def add_argument(parser):

    # training and testing names
    dataset_name = 'chess'
    parser.add_argument('--training_name', default=dataset_name, help='name of training', action='store')
    parser.add_argument('--training_name_suffix', default='', help='name of training', action='store')

    # IO file locations
    parser.add_argument('--output_data_dir', default='model/{}/'.format(dataset_name), help='location of outputs', action='store')
    parser.add_argument('--input_data_dir', default='data/{}/tfrecord2/'.format(dataset_name), help='location of training tfrecords file', action='store')
    parser.add_argument('--training_data', default=['train'], help=' list fo training tfrecord file', action='store')
    parser.add_argument('--validation_data', default=['test'], help='location of validation h5py file', action='store')
    parser.add_argument('--testing_data', default=['test'], help='location of testing h5py file', action='store')
    parser.add_argument('--pretrained_model', default='model/posenet.npy', help='pretrained googlenet model on places dataset', action='store')

    # hyper paramters
    parser.add_argument('--batch_size', default=16, help='num of samples in a batch', type=int, action='store')
    parser.add_argument('--valid_size', default=512, help='num of samples in a batch', type=int, action='store')
    parser.add_argument('--num_epochs', default=300, help='num of epochs to train', type=int, action='store')
    parser.add_argument('--lr', default=0.0001, help='initial learning rate', type=float, action='store')
    parser.add_argument('--beta', default=1, help='beta value', type=int, action='store')
    parser.add_argument('--GPU_FLAG', default='gpu01', help='which GPU machine to use', action='store' )
    parser.add_argument('--MULTIPLE_GPU_GLAG', default=False, help='which GPU machine to use', action='store' )

    # traning info file, check point and model saving location
    parser.add_argument('--tensorboard_event_logs', default='logs/tensorboard_logs/', help='tensorboard event logs', action='store')
    parser.add_argument('--csv_logs', default='logs/csv_logs/', help='csv logs', action='store')
    parser.add_argument('--checkpoint_weights', default='weights/checkpoint_weights/', help='model checkpoint storage location', action='store')
    parser.add_argument('--weights', default='weights/', help='final weight location storage', action='store')
    parser.add_argument('--training_data_info', default='training_data_info/', help='stores information such as hyper parameter, model and data location etc', action='store')

    # Other flags
    ### by default regular training and testing will run which is estimate 7 parameters (xyz, wpqr)
    parser.add_argument('--diff_location', default=False, help='test model with location specified and not from the location in training_data_info.h5py', action='store')
    parser.add_argument('--threshold_angle', default=20, help='Threshold of angle to segment image pairs', action='store')
    parser.add_argument('--threshold_dist', default=0.4, help='Threshold of angle to segment image pairs', action='store')
        
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    print("flags argument: {}".format(args))
    train = Train(args)
    train.train()



