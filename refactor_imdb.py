import os
import torch

from scipy import io
from PIL import Image
from random import shuffle
from torchvision.transforms import ToTensor, Resize, Compose, Lambda
# structure of "imdb.mat":
    #   dob -> date of birth
    #   photo_taken -> date photo was taken
    #   gender -> m/f
    #   full_path -> path to the photo
    #   others: <useless>   


import scipy.io
import numpy as np
import datetime

from utils import *



def reformat_date(mat_date):
    """ Extract only the year.
        Necessary for calculating the age of the individual in the image.
    Args:
        mat_date - raw date format.
    Retrurns:
        dt - adjusted date.
    """
    # Take account for difference in convention between matlab and python.
    dt = datetime.date.fromordinal(np.max([mat_date - 366, 1])).year
    return dt


def matlab_to_numpy(path_to_meta, matlab_file, path_to_images):
    """ Opens .mat file and reformats.
        Matlab struct format to dictionary of numpy arrays.
    Args:
        path_to_meta - path to dir with matlab meta file.
        matlab_file - matlab file
        path_to_images - incomplete paths to images.
    Returns:
        imdb_dict - dict of numpy arrays.
    """
    mat_struct = io.loadmat(os.path.join(path_to_meta, matlab_file))
    data_set = [data[0] for data in mat_struct['imdb'][0, 0]]

    keys = [
        'dob',
        'photo_taken',
        'full_path',
        'gender',
        'name',
        'face_location',
        'face_score',
        'second_face_score',
        'celeb_names',
        'celeb_id'
    ]

    # Creates path to full path to image from incomplete path.
    create_path = lambda path: os.path.normpath(os.path.join(path_to_images, path[0]))

    imdb_dict = dict(zip(keys, np.asarray(data_set)))
    imdb_dict['dob'] = [reformat_date(dob) for dob in imdb_dict['dob']]
    imdb_dict['full_path'] = [create_path(path) for path in imdb_dict['full_path']]

    # Add 'age' key to the dictionary
    imdb_dict['age'] = imdb_dict['photo_taken'] - imdb_dict['dob']

    return imdb_dict

def main():
    imdb_dict = matlab_to_numpy('imdb/imdb_crop', 'imdb.mat', 'imdb/imdb_crop')


    def age_for_imdb(path):
        # here path is supposed the full relative path to image
        i = 0
        # provare a trasformare in dict
        l = list(imdb_dict['full_path'])
        idx = l.index(path)
        return imdb_dict['age'][idx]

    os.mkdir('imdb/imdb_refined')

    it = os.walk('imdb/imdb_crop')
    _ = next(it)

    for d, _, fnames in it:
        tmp = False
        for f in fnames:
            path = os.path.normpath(os.path.join(d, f))
            if tmp: 
                print(path)
                os.system("pause")

            age = age_for_imdb(path)
            agedir = os.path.normpath(os.path.join('imdb/imdb_refined', str(age)))
            new_path = os.path.join(agedir, f)
            curr_dirs = next(os.walk('imdb/imdb_refined'))[1]
            if not agedir.split('\\')[-1] in curr_dirs:
                os.mkdir(agedir)
            shutil.copyfile(path, new_path)
        print("finished dir: ", d)

if __name__ == '__main__':
    main()
    
