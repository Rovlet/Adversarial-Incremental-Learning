from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils import data
import torch
import PIL
import torchvision.transforms as transforms
from settings import config, DATABASE, VALIDATION_SIZE
import numpy as np
from torch.utils.data import Dataset
from sklearn.utils import shuffle


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


def read_train_data(tst_transform=None, trn_transform=None):
    train_images, train_labels, validation_images, validation_labels, test_images, test_labels = read_data_sets(trn_transform=trn_transform, tst_transform=tst_transform)
    train_images, train_labels = shuffle(train_images, train_labels)
    return train_images, train_labels, validation_images, validation_labels


def read_test_data(train_dir='./data', one_hot=False, tst_transform=None):
    test_images = np.load(train_dir + f"/{config['x_test']}")
    test_labels = np.load(train_dir + f"/{config['y_test']}")
    test_images = test_images.astype(np.float32)
    if tst_transform:
        test_images = [tst_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in test_images]
    return test_images, test_labels


def read_data_sets(train_dir='./data', trn_transform=None, tst_transform=None):
    train_images = np.load(train_dir + f"/{config['x_train']}")
    train_labels = np.load(train_dir + f"/{config['y_train']}")
    test_images = np.load(train_dir + f"/{config['x_test']}")
    test_labels = np.load(train_dir + f"/{config['y_test']}")
    test_images = test_images.astype(np.float32)
    train_images = train_images.astype(np.float32)
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    if trn_transform and tst_transform:
        if DATABASE == 'cicids':
            train_images = [trn_transform(np.swapaxes(image, 0, 1).astype(np.uint8)) for image in train_images]
            test_images = [trn_transform(np.swapaxes(image, 0, 1).astype(np.uint8)) for image in test_images]
            validation_images = [trn_transform(np.swapaxes(image, 0, 1).astype(np.uint8)) for image in validation_images]
        else:
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
    #
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
    pin_memory = True

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
    if DATABASE == 'cicids':
        dc = {
            'extend_channel': None,
            'pad': None,
            'normalize': ((0.1,), (0.2752,)),
            'resize': None,
            'crop': None,
            'flip': None,
        }
    else:
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


