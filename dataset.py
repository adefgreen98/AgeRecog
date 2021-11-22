import datetime
import numpy as np
from scipy import io

import torch, torchvision
from torchvision.datasets.folder import *

__accessible_datasets__ = {
    'fgnet': ("FGNET", "FGNet_dataset/"),
    'utkface': ("UTKFace", "UTK_face_dataset/"),
    'imdb': ("imdb_refined", "imdb/")
}


def get_dataset_name(name):
    if not (name in __accessible_datasets__.keys()):
        raise ValueError("cannot find dataset with name '{}'".format(name))
    else:
        return __accessible_datasets__[name][1] + __accessible_datasets__[name][0]


class ReducedDataset(torchvision.datasets.VisionDataset):
    """ Dataset used to limit maximum age taken in consideration; most of the code comes from 'torchvision.datasets.DatasetFolder'."""
    # Not used in final thesis
    
    def __init__(self, root, reduction_function, loader=default_loader, extensions=('.jpg', '.jpeg', '.png'), transforms=None, transform=None, target_transform=None, is_valid_file=None):
        super(ReducedDataset, self).__init__(root, transform=transform, target_transform=target_transform)

        
        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        classes = list(filter(reduction_function, classes))
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    