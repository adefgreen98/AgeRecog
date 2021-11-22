
import os, time
import shutil
import torch, torchvision
from numpy import round

import random
import matplotlib.pyplot as plt
import seaborn as sns

 
def age_from_path(path):
    return int(path.split("_")[0])

def extract_fgnet_data(path):
    # typical structure is .../.../001A10.jpg, but it may contain a letter like '16b.jpeg'
    try:
        res = int(path[4:-4])
    except ValueError:
        res = int(path[4:-5])
    return res
    
 
def build_dataset(topdir, age_fn = age_from_path):
    """ Extracts age data from path and puts sample in the correct directory."""
    fnames = next(os.walk(topdir))[2]
 
    for el in fnames:
        if not os.path.exists(topdir + '/' + str(age_fn(el))):
            os.mkdir(topdir + '/' + str(age_fn(el)))
        
        shutil.move(topdir + '/' + el, topdir + '/' + str(age_fn(el)))
 


def create_custom_classes(classes=None):
    if classes == None:
        return [it for it in range(0, 12)]
 

def manage_targets(y, classes=None):
    """ Transforms targets when performing classification. If classes are not given, then it is considered as regression and no transformation is applied."""
    ret = y
    m = int(len(classes) * y / 103)
    if classes != None:
        i=0
        while classes[i] < m and i < len(classes) - 1:
            i += 1
        ret = i

    return ret


def get_model_fname(dir):
    fname = [el for el in next(os.walk(dir))[2] if el.endswith('.pth')]
    if len(fname) > 1:
        raise RuntimeError('cannot decide within multiple models in {}'.format(dir))
    return fname[0]


def get_class_distribution(dataset_obj):
    count_dict = {}
    for element in dataset_obj:
        y_lbl = element[1]
        if y_lbl not in count_dict.keys():
            count_dict[y_lbl] = 0
        count_dict[y_lbl] += 1
            
    return count_dict

def plot_class_distribution(dataset):
    count = get_class_distribution(dataset)
    x, y = list(count.keys()), list(count.values())
    plt.figure(figsize=(30, 10))

    ax = sns.barplot(x, y, color='royalblue')

    ax.tick_params(axis='both', labelsize=12)
    ax.tick_params(axis='x', labelrotation=45)

    plt.xlabel("Età", fontsize=16)
    plt.ylabel("Numero immagini", fontsize=16)

    plt.title("Distribuzione UTKFace", fontsize=18)
    plt.savefig('utk_distribution.png')


############### Sampler for balancing sampling from dataset ###############
# Not used in final thesis

class BalancedSampler(torch.utils.data.Sampler):
    # Readaptated from https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py

    def __init__(self, dataset):
        self.indices = list(range(len(dataset)))
        self.num_samples = len(dataset)
        labels = {}

        for i in self.indices:
            label = self.__get_label(dataset, i)
            if not (label in labels):
                labels[label] = 0
            labels[label] += 1
        
        weights = [
            1 / labels[self.__get_label(dataset, idx)] for idx in self.indices
        ]

        self.weights = torch.tensor(weights)

    def __get_label(self, dataset, index):
        # if dataset is of kind ImageFolder should be correct
        return dataset[index][1]
    
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))
    
    def __len__(self):
        return self.num_samples


############### Logging utilities ###############


def make_log_dir(topdir='', toprint=True):
    if not os.path.exists(topdir):
        os.mkdir(topdir)
        print('Making top dir for logs...') if toprint else None
    if not os.path.exists(topdir + 'logs'):
        os.mkdir(topdir + 'logs')
        print("Making log directory...") if toprint else None
    dirname = topdir + "logs/" + time.strftime("%x").replace('/', '_')
    if not os.path.exists(dirname):
        print("Making log subdirectory...") if toprint else None
        os.mkdir(dirname)
    print("Directory '{}' made for logs".format(dirname)) if toprint else None
    dirname = dirname + '/' + time.strftime("%X").replace(':', '_')
    os.mkdir(dirname)
    print("Current log is: {}".format(dirname))
    os.mkdir(dirname + '/images')
    return dirname


def make_logfile_name(logdir, toprint=True):
    name = logdir + "/" + time.strftime("%X").replace(':', '_') + ".txt"
    print("Log name: '{}'".format(name)) if toprint else None
    return name


class logger():

    log_name = ''
    log_dir = ''

    def __init__(self, params=None, topdir='.', retrain=False):
        logger.log_dir = make_log_dir(topdir  + '/')
        logger.log_name = make_logfile_name(logger.log_dir)
        logger.images_dir = logger.log_dir + '/images/'
        if params != None:
            assert type(params) == dict
            self.write(**params)
    
    def write(self, *args, **kwargs):
        """ Writes parameters or messages in log file in a fancy way. Messages are expected as arguments while parameters are expected
        as keyword arguments; eventually the former can be passed as a dictionary with syntax '**params'. If 'toprint' is set to true, 
        the result string(s) will be written also in output console. If 'plain' is set to true, the output will be printed normally"""
        lf = open(logger.log_name, mode='at')
        toprint = kwargs.pop('toprint', False)
        plain = kwargs.pop('plain', False)

        if plain:
            lf.write(''.join(args) + '\n')
            if toprint:
                print(''.join(args))
        else:    
            s = None
            if len(args) != 0:
                for el in args:
                    s = '\n---------------------------------\n' + str(el) + '\n'
                    lf.write(s)
                    if toprint: print(s)
            if len(kwargs) != 0:
                for k in kwargs:
                    s = '\n---------------------------------\n' + str(k) + ": "
                    if isinstance(kwargs[k], list) or isinstance(kwargs[k], tuple):
                        s += '('
                        for i, el in enumerate(kwargs[k]):
                            s += "  {}".format(el)
                            if i != (len(kwargs[k]) - 1):
                                s += ','
                        s += ')'
                    elif isinstance(kwargs[k], dict):
                        for i, el in enumerate(kwargs[k]):
                            s += "  {}: {}  ".format(el, kwargs[k][el])
                            if i != (len(kwargs[k]) - 1):
                                s += '|'
                    else:
                        s += str(kwargs[k])
                    s += '\n'   
                    lf.write(s)
                    if toprint: 
                        print(s)
            
        lf.flush()
        lf.close()
        return
    
    def name(self):
        return logger.log_dir.split('/')[-1]

 
################## Classification metrics ##################
 
class Metrics():
    # (i,j) -> number of samples of label <i> predicted of class <j>
    
    def __init__(self, classes=100, cs_n=5, class_names=None, mode='classification'):
        super().__init__()
        assert mode in ['classification', 'regression']
        self._conf_mat = torch.zeros(size=(classes, classes), requires_grad=False)
        self._mae_mat = torch.zeros(size=(classes, classes), requires_grad=False)
        self._classes = classes
        self.cs_n = cs_n
        self.mode = mode
        if class_names == None:
            self.class_names = range(classes)
        else:
            self.class_names = class_names
    
    def clear(self):
        self._conf_mat.zero_()
        self._mae_mat.zero_()
    
    def mat_conf(self):
        return self._conf_mat
    
    def mat_mae(self):
        return self._mae_mat
 
    def add(self, x, y):
        # Expects  1d vector of predictions and 1d vector of labels.
        # Automatically resizes target to be in range [0, num_classes] (else there would be issues
        # when using the clustered version)
        x = x.int()
        y = y.int()
 
        diff = torch.abs(x - y).to(self._mae_mat.device)
 
        # If x is outside boundaries [0, numclasses] then it is set to the
        # nearest class, that is 0 or numclass
        for i in range(len(x)): 
            if x[i] < 0:  x[i] = 0
            elif x[i] >= self.classes_nr():  x[i] = (self.classes_nr() - 1)
 
        for i in range(len(x)):
            self._conf_mat[y[i], x[i]] += 1
            self._mae_mat[y[i], x[i]] += diff[i]
 
    def acc(self):
        # zeros the lines where there have not been samples (are all zeros --> mean is nan)
        self._conf_mat[torch.isnan(self._conf_mat)] = 0
        return torch.sum(self._conf_mat.diag()) / torch.sum(self._conf_mat).item()
    
    def mae(self):
        """ Mean Absolute Error on difference between real age and prediction. """
        # zeros the lines where there have not been samples (are all zeros --> mean is nan)
        self._mae_mat[torch.isnan(self._mae_mat)] = 0
        return torch.sum(self._mae_mat) / torch.sum(self._conf_mat)
    
    def classes_nr(self):
        return self._classes
    
    def CS(self):

        # zeros the lines where there has not been added any sample (are all zeros --> mean is nan)
        self._conf_mat[torch.isnan(self._conf_mat)] = 0
        return self.cs_n, torch.sum(torch.tensor([torch.sum(self._conf_mat.diagonal(offset=i)) for i in range(-self.cs_n, self.cs_n+1)])) / torch.sum(self._conf_mat)
    
    def plot(self):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
            cm (array, shape = [n, n]): a confusion matrix of integer classes
            class_names (array, shape = [n]): String names of the integer classes
        """
        import matplotlib.pyplot as plt
        import itertools
        import numpy as np

        cm = self._conf_mat.numpy()

        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)    
        
        # zeros the lines where there have not been samples (are all zeros --> mean is nan)
        cm[np.isnan(cm)] = 0

        if self.mode == 'classification':
            tick_marks = np.arange(len(self.class_names))    
            figure = plt.figure(figsize=(8,8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion matrix")
            plt.colorbar()
            
            plt.xticks(tick_marks, self.class_names, rotation=45)
            plt.yticks(tick_marks, self.class_names)


            # Use white text if squares are dark; otherwise black.
            threshold = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                color = "white" if cm[i, j] > threshold else "black"
                plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

            # plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            return figure
        
        elif self.mode == 'regression':
            
            figure = plt.figure(figsize=(8,8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion matrix")
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            return figure
