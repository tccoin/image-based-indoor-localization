from __future__ import print_function

"""
Created on Wed Aug 29
@author: mitesh patel
read kitti sequences and store them as tfrecords
"""

import os
import cv2
import time
import numpy as np
import tensorflow as tf
import threading
import argparse
from tqdm import tqdm

network_input_image_height = 224
network_input_image_width = 224


class DataSource(object):
    def __init__(self, images, poses):
        self.images = images
        self.poses = poses


def parse_text_file_quaternion(text_file, image_file_dir):
    '''Get image names and poses from text file.'''
    poses = []
    images = []

    with open(text_file) as f:
        for line in f:
            fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
            pose = np.empty((7,), dtype=np.float32)
            # as we are doing classification we only care about the first parameter which is the label
            pose[0] = float(p0)
            pose[1] = float(p1)
            pose[2] = float(p2)
            pose[3] = float(p3)
            pose[4] = float(p4)
            pose[5] = float(p5)
            pose[6] = float(p6)
            poses.append(pose)
            images.append(image_file_dir + fname)
    return images, poses

def parse_text_file(text_file, image_file_dir):
    '''Get image names and poses from text file.'''
    poses = []
    images = []

    with open(text_file) as f:
        next(f)  # skip the 3 header lines
        next(f)
        next(f)
        for line in f:
            fname, t0, t1, t2, t3, \
                   t4, t5, t6, t7, \
                   t8, t9, t10, t11, \
                   t12, t13, t14, t15 = line.split()

            pose = np.empty((16,), dtype=np.float32)
            pose[0] = float(t0)
            pose[1] = float(t1)
            pose[2] = float(t2)
            pose[3] = float(t3)
            pose[4] = float(t4)
            pose[5] = float(t5)
            pose[6] = float(t6)
            pose[7] = float(t7)
            pose[8] = float(t8)
            pose[9] = float(t9)
            pose[10] = float(t10)
            pose[11] = float(t11)
            pose[12] = float(t12)
            pose[13] = float(t13)
            pose[14] = float(t14)
            pose[15] = float(t15)

            poses.append(pose)
            images.append(os.path.join(image_file_dir, fname))

    # return DataSource(images, poses)
    return images, poses

def writing(text_file, image_file_dir, tfrecords_filename, dataset_name, sample_dataset = False):

    # data.image contains image info and data.label contains label info
    # Training
    version = tf.__version__
    print('version: {} and type: {}: {}'.format(version, type(version), int(version.split('.')[0])))
    if (int(version.split('.')[0]) == 1 ):
        print('TENSORFLOW 1.0.0 SELECTED')
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        writer = tf.python_io.TFRecordWriter(tfrecords_filename,options=options)
    else:
        print('TENSORFLOW 2.0.0 SELECTED')
        options = tf.compat.v1.python_io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
        writer = tf.compat.v1.python_io.TFRecordWriter(tfrecords_filename, options=options)

    # data = parse_text_file(text_file, image_file_dir)
    if dataset_name is 'kitti_dataset':
        images, poses = parse_text_file(text_file, image_file_dir)
    else:
        images, poses = parse_text_file_quaternion(text_file, image_file_dir)
    if sample_dataset:
        images = images[:1000]
        poses = poses[:1000]
    data = DataSource(images,poses)
    N = 0
    mean = np.zeros((network_input_image_height, network_input_image_width, 3), dtype=np.uint64)

    for i in tqdm(range(len(data.images))):
        image = cv2.imread(data.images[i])

        image = cv2.resize(image, (network_input_image_width, network_input_image_height), cv2.INTER_CUBIC)


        mean[:, :, 0] += image[:, :, 0]
        mean[:, :, 1] += image[:, :, 1]
        mean[:, :, 2] += image[:, :, 2]
        N += 1

        bufStr = image.tostring()
        poses = data.poses[i].tostring()

        if i == 0:
            print('poses: {}'.format(data.poses[i]))
            print('poses shape: {}'.format(len(data.poses[i])))
            print('image shape: {}'.format(image.shape))
            print('buffer: {}'.format(len(bufStr)))

        # convert into same type as image, i.e. np.unit8
        # mean = np.asarray(mean / N + 0.49, dtype = np.uint8)
        # decided to keep the mean image float
        mean = np.asarray(mean, dtype=np.float64) / N


        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=
                                      tf.train.Int64List(value=[network_input_image_height])),
            'width': tf.train.Feature(int64_list=
                                     tf.train.Int64List(value=[network_input_image_width])),
            'depth': tf.train.Feature(int64_list=
                                     tf.train.Int64List(value=[3])),
            'poses': tf.train.Feature(bytes_list=
                                      tf.train.BytesList(value=[poses])),
            'compressed_image': tf.train.Feature(bytes_list=
                                      tf.train.BytesList(value=[bufStr]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

    print('\nWritten %d images to %s' % (int(len(data.images)), tfrecords_filename))

def create_joint_tfrecord_file(src_filename_list, image_file_dir, dataset_name, dest_tfrecords_filename, sample_data=False):
    print(src_filename_list)
    print(image_file_dir)
    print(dest_tfrecords_filename)

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(dest_tfrecords_filename, options=options)
    images = []
    poses = []
    dataset_len = []
    for idx, text_file in enumerate(src_filename_list):
        print(text_file)
        # img, pose = parse_text_file(text_file, image_file_dir)
        if dataset_name is 'kitti_dataset':
            img, pose = parse_text_file(text_file, image_file_dir)
        else:
            img, pose = parse_text_file_quaternion(text_file, image_file_dir)

        dataset_len.append(len(img))
        images = images + img
        poses = poses + pose
    if sample_data:
        images = images[:1000]
        poses = poses[:1000]
    data = DataSource(images, poses)
    N = 0
    mean = np.zeros((network_input_image_height, network_input_image_width, 3), dtype=np.uint64)
    print(len(data.poses))
    print('dataset lengths: {}'.format(dataset_len))
    dataset_len = np.asarray(dataset_len)
    for i in tqdm(range(len(data.images))):
        image = cv2.imread(data.images[i])
        image = cv2.resize(image, (network_input_image_width, network_input_image_height), cv2.INTER_CUBIC)


        mean[:, :, 0] += image[:, :, 0]
        mean[:, :, 1] += image[:, :, 1]
        mean[:, :, 2] += image[:, :, 2]
        N += 1

        bufStr = image.tostring()
        poses = data.poses[i].tostring()
        dataset_lens = dataset_len.tostring()

        if i == 0:
            print('poses: {}'.format(data.poses[i]))
            print('poses shape: {}'.format(len(data.poses[i])))
            print('image shape: {}'.format(image.shape))
            print('buffer: {}'.format(len(bufStr)))

        # convert into same type as image, i.e. np.unit8
        # mean = np.asarray(mean / N + 0.49, dtype = np.uint8)
        # decided to keep the mean image float
        mean = np.asarray(mean, dtype=np.float64) / N


        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=
                                      tf.train.Int64List(value=[network_input_image_height])),
            'width': tf.train.Feature(int64_list=
                                     tf.train.Int64List(value=[network_input_image_width])),
            'depth': tf.train.Feature(int64_list=
                                     tf.train.Int64List(value=[3])),
            'poses': tf.train.Feature(bytes_list=
                                      tf.train.BytesList(value=[poses])),
            'compressed_image': tf.train.Feature(bytes_list=
                                      tf.train.BytesList(value=[bufStr])),
            'dataset_lengths': tf.train.Feature(bytes_list=
                                      tf.train.BytesList(value=[dataset_lens]))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    print('\nWritten %d images to %s' % (int(len(data.images)), dest_tfrecords_filename))




if __name__ == '__main__':
    aparse = argparse.ArgumentParser(
        prog="convert_kitti data to tf records",
        description="todo ..")
    aparse.add_argument('--num_of_camera',
                        default=1)
    aparse.add_argument('--dataset',
                        action='store',
                        dest='dataset',
                        default='fxpal_dataset')
    aparse.add_argument('--max_num_threads',
                        action='store',
                        dest='max_num_threads',
                        default=4)
    aparse.add_argument('--combine_dataset',
                        action='store',
                        dest='combine_dataset',
                        default=0)
    aparse.add_argument('--sample_dataset',
                        action='store',
                        dest='sample_dataset',
                        default=0)

    cmdargs = aparse.parse_args()
    # save_dir = cmdargs.kitti_save_dir
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    max_num_threads = cmdargs.max_num_threads
    threads = []
    index = 0

    if cmdargs.dataset is 'kitti_dataset':
        num_seq = 11
        if int(cmdargs.num_of_camera) == 2:
            cam_list = ['image_2', 'image_3']
        else:
            cam_list = list(['image_2'])
    else:
        num_seq = 2
        #num_seq = 25    Modified by Jingwei 19 July 2019

    if cmdargs.combine_dataset:

        if cmdargs.dataset is 'kitti_dataset':
            # Specify the video files of the training, validation and test set
            ids_train = [0, 1, 2, 8, 9]
            ids_test = [3, 5, 7]
            ids_val = [4, 6, 10]
            ids = [ids_train, ids_test, ids_val]

            dataset_dir = '/ssd/data/kitti_dataset/'
            save_dir = '/ssd/data/kitti_dataset/tfrecords2/'
            image_file_dir = dataset_dir

            name_train = 'kitti_joint_training.tfrecords'
            name_test = 'kitti_joint_testing.tfrecords'
            name_val = 'kitti_joint_validation.tfrecords'
            image_file_dir = cmdargs.kitti_dataset_dir
            file_names = [name_train, name_test, name_val]

            index = 0

            create_joint_tfrecord_file(
                [os.path.join(cmdargs.kitti_dataset_dir, 'formatted_' + cam + '_' + ('%02d.txt' % i)) for i in ids_train
                 for cam in cam_list],image_file_dir, cmdargs.dataset, os.path.join(cmdargs.kitti_dataset_dir, name_train)
            )
            create_joint_tfrecord_file(
                [os.path.join(cmdargs.kitti_dataset_dir, 'formatted_' + cam + '_' + ('%02d.txt' % i)) for i in ids_val
                 for cam in cam_list], image_file_dir, cmdargs.dataset,
                os.path.join(cmdargs.kitti_dataset_dir, name_val)
            )
            create_joint_tfrecord_file(
                [os.path.join(cmdargs.kitti_dataset_dir, 'formatted_' + cam + '_' + ('%02d.txt' % i)) for i in ids_test
                 for cam in cam_list], image_file_dir, cmdargs.dataset,
                os.path.join(cmdargs.kitti_dataset_dir, name_test)
            )

        else:
            #  Modified by Jingwei 3 August
            ids_train = [0]
            ids_test = [1]
            ids_val = [0]
            ids = [ids_train, ids_test, ids_val]
            #dataset_dir = '/ssd/data/fxpal_dataset/'
            #save_dir = '/ssd/data/fxpal_dataset/posenetDataset/tfrecord2/'
            dataset_dir = '/media/jingwei/Fujisu/dataset/NVIDIA/indoor_dataset/warehouse_dark/'
            save_dir = '/media/jingwei/Fujisu/dataset/NVIDIA/indoor_dataset/warehouse_dark/posenetDataset/tfrecord2/'
            image_file_dir = dataset_dir + 'sequences/'

            name_train = 'training.tfrecords'
            name_test = 'testing.tfrecords'
            name_val = 'validation.tfrecords'
            file_names = [name_train, name_test, name_val]

            create_joint_tfrecord_file(
                [os.path.join(dataset_dir, ('%02d.txt' % i)) for i in ids_train],image_file_dir, cmdargs.dataset, os.path.join(save_dir, name_train) )
            create_joint_tfrecord_file(
                [os.path.join(dataset_dir, ('%02d.txt' % i)) for i in ids_test], image_file_dir, cmdargs.dataset, os.path.join(save_dir, name_test))
            create_joint_tfrecord_file(
                [os.path.join(dataset_dir, ('%02d.txt' % i)) for i in ids_val], image_file_dir, cmdargs.dataset, os.path.join(save_dir, name_val))


    elif cmdargs.sample_dataset:
        ids_train = 0
        dataset_dir = '/ssd/data/fxpal_dataset/'
        save_dir = '/ssd/data/fxpal_dataset/posenetDataset/tfrecord2/'
        image_file_dir = dataset_dir + 'sequences/'

        text_file = os.path.join(dataset_dir, ('%02d.txt' % ids_train))
        tfrecords_filename = os.path.join(save_dir, 'sample_dataset.tfrecords')

        writing(text_file, image_file_dir, tfrecords_filename, cmdargs.dataset, sample_dataset = True)

    else:

        while True:
            threads = [t for t in threads if t.is_alive()]

            # add thread
            #   Modified by Jingwei, 11 Sep 2019
            if len(threads) < max_num_threads and index < num_seq:
                if cmdargs.dataset is 'kitti_dataset':
                    dataset_dir = '/ssd/data/kitti_dataset/'
                    save_dir = '/ssd/data/kitti_dataset/tfrecords2/'
                    text_file = os.path.join(dataset_dir, 'formatted_' + cam_list[0] + '_' + ('%02d.txt' % index))
                    image_file_dir = dataset_dir
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    tfrecords_filename = os.path.join(save_dir, 'formatted_' + cam_list[0] + '_' + ('%d.tfrecords' % index))
                else:
                    #   Modified by Jingwei, 19 July 2019
                    dataset_dir = '/media/jingwei/Lenovo/dataset/NVIDIA/indoor_dataset/warehouse_dark/'
                    save_dir = '/media/jingwei/Lenovo/dataset/NVIDIA/indoor_dataset/warehouse_dark/posenetDataset/tfrecord2/'
                    image_file_dir = dataset_dir + 'sequences/' + ('%02d' % index) + '/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    text_file = os.path.join(dataset_dir, ('%02d.txt' % index))
                    tfrecords_filename = os.path.join(save_dir, ('%d.tfrecords' % index))

                print('text file location: {}'.format(text_file))
                print('image file location: {}'.format(image_file_dir))
                # tfrecords_filename = os.path.join(save_dir, '%d.tfrecords' % index)
                t = threading.Thread(target=writing, args=(text_file, image_file_dir, tfrecords_filename, cmdargs.dataset))
                threads.append(t)
                t.start()
                print("Thread for ", tfrecords_filename, " is added.")

                index += 1

            if index == num_seq:
                break

            time.sleep(1)


        for t in threads:
            t.join()
