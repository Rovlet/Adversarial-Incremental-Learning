from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils import data
import torch
import PIL
import torchvision.transforms as transforms
import urllib

import gzip
import os
import numpy as np
from six.moves import urllib
from torch.utils.data import Dataset
from sklearn.utils import shuffle


def maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


class BaseDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all paths in memory"""

    def __init__(self, dataset, class_indices=None):
        """Initialization"""
        # self.transform = transform
        self.labels = dataset['y']
        self.images = dataset['x']
        # self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.images[index]
        y = self.labels[index]
        y = torch.tensor(y, dtype=torch.long)
        return x, y


def read_train_data(train_dir='./data', one_hot=False, VALIDATION_SIZE=0, tst_transform=None, trn_transform=None):
    train_images, train_labels, validation_images, validation_labels, test_images, test_labels = read_data_sets(trn_transform=trn_transform, tst_transform=tst_transform)
    train_images, train_labels = shuffle(train_images, train_labels)
    return train_images, train_labels, validation_images, validation_labels

#
def read_test_data(train_dir='./data', one_hot=False, tst_transform=None):
    # TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    # TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    # local_file = maybe_download(TEST_IMAGES, train_dir)
    # test_images = extract_images(local_file)

    # local_file = maybe_download(TEST_LABELS, train_dir)
    # test_labels = extract_labels(local_file, one_hot=one_hot)
    test_images = np.load(train_dir + '/test_images.npy')
    test_labels = np.load(train_dir + '/test_labels.npy')
    test_images = test_images.astype(np.float32)



    # TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    # TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    # local_file = maybe_download(TEST_IMAGES, train_dir)
    # test_images = extract_images(local_file)
    # test_images = test_images.astype(np.float32)
    # local_file = maybe_download(TEST_LABELS, train_dir)
    # test_labels = extract_labels(local_file, one_hot=one_hot)
    if tst_transform:
        test_images = [tst_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in test_images]
    return test_images, test_labels
#
#
# def read_data_sets(train_dir='./data', fake_data=False, one_hot=False, trn_transform=None, tst_transform=None):
#     TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
#     TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
#     TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
#     TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
#     VALIDATION_SIZE = 2000
#     local_file = maybe_download(TRAIN_IMAGES, train_dir)
#     train_images = extract_images(local_file)
#     local_file = maybe_download(TRAIN_LABELS, train_dir)
#     train_labels = extract_labels(local_file, one_hot=one_hot)
#     local_file = maybe_download(TEST_IMAGES, train_dir)
#     test_images = extract_images(local_file)
#     test_images = test_images.astype(np.float32)
#     local_file = maybe_download(TEST_LABELS, train_dir)
#     test_labels = extract_labels(local_file, one_hot=one_hot)
#     validation_images = train_images[:VALIDATION_SIZE]
#     validation_labels = train_labels[:VALIDATION_SIZE]
#     train_images = train_images[VALIDATION_SIZE:]
#     train_labels = train_labels[VALIDATION_SIZE:]
#     if trn_transform and tst_transform:
#         train_images = [trn_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in train_images]
#         test_images = [tst_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in test_images]
#         validation_images = [tst_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in validation_images]
#     return train_images, train_labels, validation_images, validation_labels, test_images, test_labels



#
# def read_test_data(train_dir='./data', one_hot=False, tst_transform=None):
#     TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
#     TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
#     local_file = maybe_download(TEST_IMAGES, train_dir)
#     test_images = extract_images(local_file)
#     test_images = test_images.astype(np.float32)
#     local_file = maybe_download(TEST_LABELS, train_dir)
#     test_labels = extract_labels(local_file, one_hot=one_hot)
#     if tst_transform:
#         test_images = [tst_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in test_images]
#     return test_images, test_labels
# #
# #
# def read_data_sets(train_dir='./data', fake_data=False, one_hot=False, trn_transform=None, tst_transform=None):
#     TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
#     TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
#     TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
#     TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
#     VALIDATION_SIZE = 2000
#     local_file = maybe_download(TRAIN_IMAGES, train_dir)
#     train_images = extract_images(local_file)
#     local_file = maybe_download(TRAIN_LABELS, train_dir)
#     train_labels = extract_labels(local_file, one_hot=one_hot)
#     local_file = maybe_download(TEST_IMAGES, train_dir)
#     test_images = extract_images(local_file)
#     test_images = test_images.astype(np.float32)
#     local_file = maybe_download(TEST_LABELS, train_dir)
#     test_labels = extract_labels(local_file, one_hot=one_hot)
#     validation_images = train_images[:VALIDATION_SIZE]
#     validation_labels = train_labels[:VALIDATION_SIZE]
#     train_images = train_images[VALIDATION_SIZE:]
#     train_labels = train_labels[VALIDATION_SIZE:]
#     if trn_transform and tst_transform:
#         train_images = [trn_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in train_images]
#         test_images = [tst_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in test_images]
#         validation_images = [tst_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in validation_images]
#     return train_images, train_labels, validation_images, validation_labels, test_images, test_labels


#
def read_data_sets(train_dir='./data', fake_data=False, one_hot=False, trn_transform=None, tst_transform=None):
    # TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    # TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    # TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    # TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 2000
    # local_file = maybe_download(TRAIN_IMAGES, train_dir)
    train_images = np.load(train_dir + '/train_images.npy')
    train_labels = np.load(train_dir + '/train_labels.npy')
    test_images = np.load(train_dir + '/test_images.npy')
    test_labels = np.load(train_dir + '/test_labels.npy')

    # train_images = extract_images(local_file)
    # local_file = maybe_download(TRAIN_LABELS, train_dir)
    # train_labels = extract_labels(local_file, one_hot=one_hot)
    # local_file = maybe_download(TEST_IMAGES, train_dir)
    # test_images = extract_images(local_file)
    test_images = test_images.astype(np.float32)
    train_images = train_images.astype(np.float32)
    # local_file = maybe_download(TEST_LABELS, train_dir)
    # test_labels = extract_labels(local_file, one_hot=one_hot)
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    if trn_transform and tst_transform:
        train_images = [trn_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in train_images]
        test_images = [tst_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in test_images]
        validation_images = [tst_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in validation_images]
    return train_images, train_labels, validation_images, validation_labels, test_images, test_labels


def get_transforms(resize, pad, crop, flip, normalize, extend_channel):
    """Unpack transformations and apply to train or test splits"""

    trn_transform_list = []
    tst_transform_list = []

    # resize
    if resize is not None:
        trn_transform_list.append(transforms.Resize(resize))
        tst_transform_list.append(transforms.Resize(resize))

    # padding
    if pad is not None:
        trn_transform_list.append(transforms.Pad(pad))
        tst_transform_list.append(transforms.Pad(pad))

    # crop
    if crop is not None:
        trn_transform_list.append(transforms.RandomResizedCrop(crop))
        tst_transform_list.append(transforms.CenterCrop(crop))

    # flips
    if flip:
        trn_transform_list.append(transforms.RandomHorizontalFlip())

    # to tensor
    trn_transform_list.append(transforms.ToTensor())
    tst_transform_list.append(transforms.ToTensor())

    # normalization
    if normalize is not None:
        trn_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))
        tst_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))

    # gray to rgb
    if extend_channel is not None:
        trn_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))
        tst_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))

    return transforms.Compose(trn_transform_list), \
           transforms.Compose(tst_transform_list)


def get_loaders(datasets, nc_first_task, batch_size, num_workers, pin_memory, validation=.1):
    """Apply transformations to Datasets and create the DataLoaders for each task"""
    trn_load, val_load, tst_load = [], [], []
    pin_memory=True

    # transformations
    trn_transform, tst_transform = get_transform()

    # datasets
    train_images, train_labels, validation_images, validation_labels, test_images, test_labels = read_data_sets(
        trn_transform=trn_transform, tst_transform=tst_transform)

    collected_data = {}
    collected_data['name'] = 'task-0'
    collected_data['trn'] = {'x': [], 'y': []}
    collected_data['val'] = {'x': [], 'y': []}
    collected_data['tst'] = {'x': [], 'y': []}
    collected_data['trn']['x'] = train_images
    collected_data['trn']['y'] = train_labels
    collected_data['val']['x'] = validation_images
    collected_data['val']['y'] = validation_labels
    collected_data['tst']['x'] = test_images
    collected_data['tst']['y'] = test_labels

    num_classes = len(np.unique(collected_data['trn']['y']))
    class_indices = list(range(num_classes))
    Dataset = BaseDataset

    trn_dset = Dataset(collected_data['trn'], class_indices)
    val_dset = Dataset(collected_data['val'], class_indices)
    tst_dset = Dataset(collected_data['tst'], class_indices)
    # n = 0
    # collected_data['ncla'] = n
    trn_load = data.DataLoader(dataset=trn_dset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_load = data.DataLoader(dataset=val_dset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    tst_load = data.DataLoader(dataset=tst_dset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return trn_load, val_load, tst_load


def get_transform():
    dc = {
        'extend_channel': 3,
        'pad': 2,
        'normalize': ((0.1,), (0.2752,)),
        'resize': None,
        'crop': None,
        'flip': None,
    }
    trn_transform, tst_transform = get_transforms(resize=dc['resize'],
                                                  pad=dc['pad'],
                                                  crop=dc['crop'],
                                                  flip=dc['flip'],
                                                  normalize=dc['normalize'],
                                                  extend_channel=dc['extend_channel'])
    return trn_transform, tst_transform


def get_adv_loaders(adv_images, adv_labels, task, train_perc=0.75, val_perc=0.15, batch_size=50):
    adv_images = [torch.from_numpy(image).float() for image in adv_images]
    collected_data = {}
    collected_data['name'] = 'task-' + str(task)
    collected_data['trn'] = {'x': adv_images[:int(train_perc * len(adv_images))], 'y': adv_labels[:int(train_perc * len(adv_labels))]}
    collected_data['val'] = {'x': adv_images[int(train_perc * len(adv_images)):int((train_perc + val_perc) * len(adv_images))], 'y': adv_labels[int(train_perc * len(adv_labels)):int((train_perc + val_perc) * len(adv_labels))]}
    collected_data['tst'] = {'x': adv_images[int((train_perc + val_perc) * len(adv_images)):], 'y': adv_labels[int((train_perc + val_perc) * len(adv_labels)):]}

    collected_data['ncla'] = len(np.unique(collected_data['trn']['y']))
    class_indices = {label: idx for idx, label in enumerate(np.unique(collected_data['trn']['y']))}
    Dataset = BaseDataset
    trn_dset = Dataset(collected_data['trn'], class_indices)
    val_dset = Dataset(collected_data['val'], class_indices)
    tst_dset = Dataset(collected_data['tst'], class_indices)
    trn_load = data.DataLoader(dataset=trn_dset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_load = data.DataLoader(dataset=val_dset, batch_size=batch_size, shuffle=False, pin_memory=True)
    tst_load = data.DataLoader(dataset=tst_dset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return trn_load, val_load, tst_load


