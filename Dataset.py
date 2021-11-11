from keras.datasets import mnist, fashion_mnist, cifar100, cifar10
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import skdim
import h5py
import os


class Dataset:
    '''
    This class is mainly used as a generator of objects containing the different datasets using class methods. The get_dataset_features is relevant to the dataset but is not used in correlation to any Dataset objects
    '''

    def __init__(self, data):
        self.X_train, self.y_train, self.X_test, self.y_test, self.name = data
        self.train_instances = self.X_train.shape[0]
        self.test_instances = self.X_test.shape[0]
        self.n_features = self.X_train.shape[1]

    @staticmethod
    def get_dataset_features(X_train, y_train):
        dataset_features = {}
        dataset_features['intr_dim'] = skdim.id.lPCA().fit(X_train).dimension_
        dataset_features['n_of_instances'] = X_train.shape[0]
        dataset_features['n_of_classes'] = len(np.unique(y_train))
        dataset_features['n_of_features'] = X_train.shape[1]
        dataset_features['silhouette'] = silhouette_score(X_train, y_train)
        return dataset_features

    @classmethod
    def cifar100(cls):
        '''
        Retrieves the cifar100 dataset, splits it in training and testing sets,
        reshapes it and scales its values in [0, 1] range
        '''
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()

        # reshape data to be 2D
        X_train = X_train.reshape(50000, 3072)
        X_test = X_test.reshape(10000, 3072)

        # reshape labels to 1D
        y_train = y_train.reshape(50000)
        y_test = y_test.reshape(10000)

        # ensure it's float32
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # scale values to be between 0 and 1
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        return cls([X_train, y_train, X_test, y_test, 'cifar100'])

    @classmethod
    def cifar10(cls):
        '''
        Retrieves the cifar10 dataset, splits it in training and testing sets,
        reshapes it and scales its values in [0, 1] range
        '''
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        # reshape data to be 2D
        X_train = X_train.reshape(50000, 3072)
        X_test = X_test.reshape(10000, 3072)

        # reshape labels to 1D
        y_train = y_train.reshape(50000)
        y_test = y_test.reshape(10000)

        # ensure it's float32
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # scale values to be between 0 and 1
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        return cls([X_train, y_train, X_test, y_test, 'cifar10'])

    @classmethod
    def mnist(cls):
        '''
        Retrieves the MNIST dataset, splits it in training and testing sets,
        reshapes it and scales its values in [0, 1] range
        '''
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # reshape data to be 2D
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)

        # ensure it's float32
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # scale values to be between 0 and 1
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        return cls([X_train, y_train, X_test, y_test, 'mnist'])

    @classmethod
    def fashion_mnist(cls):
        '''
        Retrieves the Fashion MNIST dataset, splits it in training and testing sets,
        reshapes it and scales its values in [0, 1] range
        '''
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        # reshape data to be 2D
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)

        # ensure it's float32
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # scale values to be between 0 and 1
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        return cls([X_train, y_train, X_test, y_test, 'fashion_mnist'])

    @classmethod
    def coil20(cls):
        '''
        Retrieves the COIL 20 dataset, splits it in training and testing sets,
        reshapes it and scales its values in [0, 1] range
        '''

        folder_path = 'datasets/coil-20-proc/'
        dataset = []
        labels = []

        # retrieve image files from folder and convert them to numpy arrays
        for obj in os.listdir(folder_path):
            if obj.endswith('.png'):
                image = Image.open(folder_path + obj)
                image_to_array = np.asarray(image).reshape(
                    16384,
                )
                dataset.append(image_to_array)
                labels.append(int(obj[3:5].replace('_', '')))
        dataset = np.array((dataset))
        # labels = np.array((labels)).reshape(len(labels), 1)
        labels = np.array((labels))

        # ensure dataset is in float32 and labels in int64
        dataset = dataset.astype('float32')
        labels = labels.astype('int64')

        # scale values to be between 0 and 1
        dataset = dataset / 255.0

        # split and shuffle the data
        X_train, X_test, y_train, y_test = train_test_split(
            dataset, labels, test_size=0.3)

        return cls([X_train, y_train, X_test, y_test, 'coil20'])

    @classmethod
    def intel_img_class(cls):
        dataset = []
        labels = []
        labels_to_num = {
            'buildings': 0,
            'forest': 1,
            'glacier': 2,
            'mountain': 3,
            'sea': 4,
            'street': 5
        }

        folder_path = 'datasets/intel_img_class/'
        for class_dir in os.listdir(folder_path):
            label = labels_to_num[class_dir]
            class_dir = folder_path + class_dir + '/'
            for obj in os.listdir(class_dir):
                if obj.endswith('.jpg'):
                    image = Image.open(class_dir + obj)
                    image_to_array = np.asarray(image)
                    if(image_to_array.shape == (150, 150, 3)):
                        image_to_array = image_to_array.reshape(67500)
                    else:
                        continue
                    dataset.append(image_to_array)
                    labels.append(label)
        dataset = np.array((dataset))
        labels = np.array((labels))

        # ensure dataset is in float32 and labels in int64
        dataset = dataset.astype('float32')
        labels = labels.astype('int64')

        # scale values to be between 0 and 1
        dataset = dataset / 255.0

        # split and shuffle the data
        X_train, X_test, y_train, y_test = train_test_split(
            dataset, labels, test_size=0.3)

        return cls([X_train, y_train, X_test, y_test, 'intel_img_class'])

    @classmethod
    def shipsnet(cls):
        dataset = []
        labels = []

        folder_path = 'datasets/shipsnet/'
        for obj in os.listdir(folder_path):
            if obj.endswith('.png'):
                image = Image.open(folder_path + obj)
                image_to_array = np.asarray(image)
                if(image_to_array.shape == (80, 80, 3)):
                    image_to_array = image_to_array.reshape(19200)
                else:
                    continue
                dataset.append(image_to_array)
                labels.append(int(obj[0]))
        dataset = np.array((dataset))
        labels = np.array((labels))

        # ensure dataset is in float32 and labels in int64
        dataset = dataset.astype('float32')
        labels = labels.astype('int64')

        # scale values to be between 0 and 1
        dataset = dataset / 255.0

        # split and shuffle the data
        X_train, X_test, y_train, y_test = train_test_split(
            dataset, labels, test_size=0.3)

        return cls([X_train, y_train, X_test, y_test, 'shipsnet'])

    @classmethod
    def usps(cls):
        f = h5py.File('datasets/usps.h5', 'r')

        train_set = f['train']
        X_train = np.array(train_set['data'])
        y_train = np.array(train_set['target'])

        test_set = f['test']
        X_test = np.array(test_set['data'])
        y_test = np.array(test_set['target'])

        # reshape labels to 1D
        y_train = y_train.reshape(7291)
        y_test = y_test.reshape(2007)

        # ensure it's float32
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        return cls([X_train, y_train, X_test, y_test, 'usps'])

    @classmethod
    def umist(cls):
        dataset = []
        labels = []

        folder_path = 'datasets/umist/'
        for class_dir in os.listdir(folder_path):
            # transform the string label to a number so that '1a' = 1 and so on
            label = ord(class_dir[1]) - 96
            class_dir = folder_path + class_dir + '/'
            for obj in os.listdir(class_dir):
                if obj.endswith('.pgm'):
                    image = Image.open(class_dir + obj)
                    image_to_array = np.asarray(image)
                    image_to_array = image_to_array.reshape(10304)
                    dataset.append(image_to_array)
                    labels.append(label)
        dataset = np.array((dataset))
        labels = np.array((labels))

        # ensure dataset is in float32 and labels in int64
        dataset = dataset.astype('float32')
        labels = labels.astype('int64')

        # scale values to be between 0 and 1
        dataset = dataset / 255.0

        # split and shuffle the data
        X_train, X_test, y_train, y_test = train_test_split(
            dataset, labels, test_size=0.3)

        return cls([X_train, y_train, X_test, y_test, 'umist'])
