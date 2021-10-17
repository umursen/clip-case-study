from torch.utils.data import Dataset
import os
import glob
from PIL import Image
from collections import Counter
import numpy as np
import math


def get_class(path):
    return path.split('/')[-2]


class CocoCaseStudyDataset(Dataset):
    def __init__(self, split, experiment_type, transform, split_ratio=None, excluded_paths=None):
        """
            Constructor for the dataset

            Parameters
            ----------
            split (str): Name of the split. Choose from train, validation, or test.
            experiment_type (str): The type of the experiment. Choose from few or zero.
            transform (Compose): Transformation to apply on data.
            split_ratio (float): The data split ratio. Calculation uses ceil, not floor. The default value is None.
            excluded_paths (list<str>): List of paths to exclude.
        """

        assert split in ['train', 'validation', 'test']
        assert experiment_type in ['few', 'zero']

        base_data_dir = 'data'

        if split == 'validation':
            split_dir = 'train'
        else:
            split_dir = split

        class_dir = os.path.join(base_data_dir, f'coco_crops_{experiment_type}_shot', split_dir)
        self._data_paths = glob.glob(class_dir+'/*/*')
        self._transform = transform

        # If provided, split ratio samples data and removes from the _data_paths list.
        if split_ratio:
            class_names = os.listdir(class_dir)
            for class_name in class_names:
                data_paths = glob.glob(os.path.join(class_dir, class_name, '*'))
                num_of_items_to_remove = math.ceil(len(data_paths)*(1.0-split_ratio))
                items_to_remove = list(np.random.choice(data_paths, num_of_items_to_remove))
                self._data_paths = list(set(self._data_paths) - set(items_to_remove))

        if excluded_paths:
            self._data_paths = list(set(self._data_paths) - set(excluded_paths))

    def __getitem__(self, index):
        path = self._data_paths[index]
        image = Image.open(path)
        image = self._transform(image)
        return image, get_class(path)

    def __len__(self):
        return len(self._data_paths)

    def get_class_counts(self):
        classes = [get_class(path) for path in self._data_paths]
        return Counter(classes)

    def get_class_names(self):
        classes = [get_class(path) for path in self._data_paths]
        return list(set(classes))

    def get_data_paths(self):
        return self._data_paths
