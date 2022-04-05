from siamese_network import posenet
from siamese_network.helper import readDataFromFile, parse_function
import tensorflow as tf

class ImageBasedLocalization():
    def __init__(self, args):

        # set attr
        self.model_info_path = 'model/{}/training_data_info/{}_posenet_data.h5py'.format(
            args.model_name,
            args.model_name
        )
        self.map_data_file = 'data/tfrecord2/train.tfrecords'
        self.test_data_file = 'data/tfrecord2/test.tfrecords'

        # init model
        self.model, self.batch_size = self.load_model()

    def load_model(self):
        data_dict = readDataFromFile(self.model_info_path)
        model =  posenet.create_siamese_keras()
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
    
    def generate_map(self):
        dataset, poses = self.load_tfrecord_dataset(self.map_data_file)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(5)
        self.map = self.model.predict(dataset)
        # todo: save map

    def load_map(self, map_path):
        pass

    def search_image(self, dataset):
        image_descriptor = self.model.predict(dataset)[0]
        # todo: search in self.map
        pass


    def run_localization(self):
        pass

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
    