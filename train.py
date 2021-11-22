import os, random, tarfile
import shutil

import torch
import torchvision
from torch.utils import tensorboard as tb

import dataset
from utils import *
from loss import *
from net import get_model, DropoutMobileNet


################## Dataset choice ##################
dataset_dir = "D:\\projects_coding\\internship\\UTK_face_dataset\\UTKFace"


parent_dataset_dir = os.path.join(dataset_dir, os.pardir)


################## Directory initialization ##################
__nr_total_images__ = -1
__nr_total_classes__ = -1

if not os.path.exists(dataset_dir):
    print("Extracting dataset from compressed archive...")

    # TODO: add possible extensions
    tf = tarfile.open(dataset_dir + ".tar.gz")

    # should automatically create a subdir with dataset_dir name (because of the extraction)
    tf.extractall(parent_dataset_dir)
    __nr_total_images__ = len(next(os.walk(dataset_dir))[2])


# checks if data has already been sorted
# in order to sort it, expects to find no folders inside dataset directory
if len(next(os.walk(dataset_dir))[1]) == 0:
    if __nr_total_images__ < 0:
        __nr_total_images__ = len(next(os.walk(dataset_dir))[2])
    print("Creating dataset directory structure...")
    build_dataset(dataset_dir, extract_fgnet_data if "FGNET" in dataset_dir else age_from_path)
else:
    print("Data has already been sorted.")

_, __total_classes__, _ = next(os.walk(dataset_dir))
if __nr_total_images__ < 0:
    __nr_total_images__ = 0
    for _, _, files in os.walk(dataset_dir):
        __nr_total_images__ += len(files)

__nr_total_classes__ = max([int(el) for el in __total_classes__]) + 1 




################## Parameters ##################


__mode__ = 'regression'

# possibility to use custom-sized classes
custom_classification=False

# Possibility to implement 3channel grayscale (also 1channel with 2classifiermn), but never used
__grayscale__ = {
    "none": None,
    "1ch": 1,
    "3ch": 3
}
__grayscale__ = __grayscale__["none"]

# list of classes 'boundaries' 
l_classes = None

# list of classes 'names' 
classes_names=None

if __mode__=='classification' and custom_classification:
    l_classes = [0, 3, 7, 12, 18, 25, 35, 50, 65, 75, 95, 120]
    classes_names = ['{} - {}'.format(l_classes[i], l_classes[i+1]) for i in range(len(l_classes) - 1)]
    __nr_total_classes__ = len(l_classes) - 1
elif __mode__ == 'regression':
    pass
elif __mode__ == 'classification' and not custom_classification:
    __nr_total_classes__ = __nr_total_classes__ // 10 + 1
    l_classes = list(range(__nr_total_classes__))
    classes_names = classes_names = ['{} - {}'.format(l_classes[i]*10, l_classes[i+1]*10) for i in range(len(l_classes) - 1)]
else:
    raise RuntimeError("invalid argument for task type : {}".format(__mode__))

params = {
    "Image size": (224, 224),
    "Dataset sizes": {'train': 100, 'valid': 50, 'test': 200},
    "Batch size": 50,
    "Epochs": 50,
    "Learning rate": 1e-3,
    "Device": 'cuda',
    "Loss function": MVLoss(),
    "Classes": '[' + (', '.join(classes_names)) + ']' if l_classes != None else ('continuous' if __mode__ == 'regression' else 'decades'),
    "Classes number": __nr_total_classes__,
    "Dropout probability": 0.2,
    "Grayscale": __grayscale__,
    "Regeneration steps": None # [75]
}
 
################## Model initialization ################## 


pretrained=True

# type of model to retrain, can be '2classifiermn', 'mobilenet', 'resnet18' or 'dropoutmn'
NET = get_model("2classifiermn", __nr_total_classes__, pretrained, params.get("Dropout probability", 0))

# stores compact information about the model
# NOTE: the part that specifies if it uses double classifier must be added manually
params["Model"] = NET._get_name() + (' (pretrained)' if pretrained and not isinstance(NET, DropoutMobileNet) else '') + " - double classifier"


################## Optimizer and Scheduler ##################

# params["Optimizer"] = torch.optim.Adam(NET.parameters(), params["Learning rate"])    

# 'Correct' version of Adam + weight decay
params['Optimizer'] = torch.optim.AdamW(NET.parameters(), params['Learning rate'], weight_decay=0.005)

# params['Optimizer'] = torch.optim.SGD(NET.parameters(), params['Learning rate'], weight_decay=0.0005, nesterov=True, momentum=0.5)

SCHED = []
# SCHED.append(torch.optim.lr_scheduler.CosineAnnealingLR(params["Optimizer"], 10))
milestones = [25, 45]
SCHED.append(torch.optim.lr_scheduler.MultiStepLR(params["Optimizer"], milestones, gamma=0.1))

if type(SCHED) == list:
    params["Scheduler"] = "".join([s.__class__.__name__ + "\n" + (str(s.state_dict()) + "\n" if s != None else "") for s in SCHED])
else: 
    params["Scheduler"] = SCHED.__class__.__name__ + "\n" + (str(SCHED.state_dict()) + "\n" if SCHED != None else "")
 



################## Log initialization ##################
 
log = logger(params, parent_dataset_dir); 

# saves the entire data in a tensorboard file (can be viewed with tensorboard dev)
# first deletes the content of './tblog' directory to avoid accumulation of multiple runs
if os.path.exists('./tblog'):
    shutil.rmtree('./tblog')
    os.mkdir('./tblog')

tblog = None
tblog_name = './tblog'; tblog = tb.SummaryWriter(tblog_name); 
 


################## Dataset initialization ##################

# decide whether to use a restricted number of ages or not
using_reduced_dataset = True
age_limit = 90

if using_reduced_dataset:
    complete_dataset = dataset.ReducedDataset(
        dataset_dir, reduction_function= (lambda x: int(x)<age_limit), 
        transform=None, target_transform=None
    )
else: 
    complete_dataset = torchvision.datasets.ImageFolder(dataset_dir, transform=None, target_transform=None)

# needed to correctly retrieve age label when loading data from directory structure
inv_dict = {v: int(k) for k, v in complete_dataset.class_to_idx.items()}
target_transform = (lambda t : manage_targets(t, l_classes)) if __mode__ != 'regression' else (lambda t: inv_dict[t])
complete_dataset.target_transform = target_transform

class TransformManager():
    """ Utility object to switch loading transformations between training mode and evaluation mode.
    In training mode data augmentation is used, while in evaluation the images are just resized.
    """

    identity_transf = torchvision.transforms.Lambda(lambda x: x)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=params["Grayscale"]) if params["Grayscale"] != None else identity_transf,
        torchvision.transforms.RandomApply(
                    [torchvision.transforms.RandomAffine(degrees=10, shear=16),
                    torchvision.transforms.RandomHorizontalFlip(p=1.0),
                    ], p=0.5),
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop(params["Image size"]),
        torchvision.transforms.Resize(params["Image size"]),
        torchvision.transforms.ToTensor()
    ])

    valid_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])


    def __init__(self, dataset):
        super().__init__()
        self.is_validation = False
        self.dataset = dataset
        self.dataset.transfom = TransformManager.train_transform
    
    def change_transformation(self):
        if self.is_validation:
            self.dataset.transform = TransformManager.valid_transform
        else:
            self.dataset.transform = TransformManager.train_transform
        self.is_validation = not self.is_validation

transform_manager = TransformManager(complete_dataset)


def regenerate_datasets(dataset=complete_dataset, params=params, sampler=None):
    indices = [i for i in range(len(dataset))]
    random.shuffle(indices)
    indices = [
        indices[0 : params["Dataset sizes"]['train']],
        indices[params["Dataset sizes"]['train'] : params["Dataset sizes"]['train'] + params["Dataset sizes"]['valid']],
        indices[params["Dataset sizes"]['train'] + params["Dataset sizes"]['valid'] : params["Dataset sizes"]['train'] + params["Dataset sizes"]['valid'] + params["Dataset sizes"]['test']]
    ]
    train_dataset = torch.utils.data.Subset(dataset, indices[0])
    valid_dataset = torch.utils.data.Subset(dataset, indices[1])
    test_dataset = torch.utils.data.Subset(dataset, indices[2])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=params["Batch size"], 
        shuffle=(sampler == None), 
        sampler=sampler
    )
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=params["Batch size"], shuffle=False)
    return train_dataset, valid_dataset, train_loader, valid_loader


train_dataset, valid_dataset, train_loader, valid_loader = regenerate_datasets(complete_dataset)


# Balanced sampler
using_balanced_sampler = False

sampler = None
if using_balanced_sampler:
    sampler = BalancedSampler(train_dataset)
    log.write("Balanced Sampler in use")




################## Training ##################

def predict(model, x):
    """ Predicts values as mean of the output distribution, when using MV-Loss as error function."""
    yp = model(x).cpu()
    # the actual prediction is the mean value of the distribution, not the max!
    prediction = torch.sum(
        torch.nn.functional.softmax(yp,dim=1) * torch.arange(__nr_total_classes__, device=yp.device), 
        dim=1
    )
    return yp, prediction

def train(model, dataloader, optimizer, loss, metrics):
    model.train()
    transform_manager.change_transformation()
    
    metrics.clear()
    
    for X,yt in dataloader:
        optimizer.zero_grad()

        # equalize device of input and model
        X = X.to(next(model.parameters()).device)

        if isinstance(loss, torch.nn.CrossEntropyLoss):
            # needed for the standard cross-entropy
            yt = yt.long()

        if isinstance(loss, MVLoss):
            yp, prediction = predict(model, X)
        else: 
            # simply the maximum
            yp = model(X)
            yp = yp.cpu()
            prediction = yp.argmax(1)
            
    
        if isinstance(loss, MVLoss):
            sfloss, mean_loss, var_loss = loss(yp, yt)
            err = sfloss + mean_loss + var_loss
        else:
            err = loss(yp, yt)
            
        err.backward()
            
        optimizer.step()
        
        metrics.add(prediction, yt)


def evaluate(model, dataloader, metrics):
    model.eval()
    transform_manager.change_transformation()

    metrics.clear()

    loss = MVLoss()

    with torch.no_grad():
        for X, yt in dataloader:

            # equalize device of input and model
            X = X.to(next(model.parameters()).device)
            
            if isinstance(loss, MVLoss):
                yp, prediction = predict(model, X)
                sfloss, mean_loss, var_loss = loss(yp, yt)
                err = sfloss + mean_loss + var_loss
            else: 
                yp = model(X).cpu()
                prediction = yp.argmax(1)
                sfloss, mean_loss, var_loss = loss(yp, yt), None, None
                err = sfloss
            
            # print("softmax loss: {} | mean loss: {} | var loss: {} \n".format(sfloss, mean_loss, var_loss))

            metrics.add(prediction, yt)


def run(model, params, mode, 
        train_dl, valid_dl, scheduler, 
        log, tblog=None, class_names=None):
    
    classes_nr = params["Classes number"]
    epochs = params["Epochs"]
    optimizer = params["Optimizer"]
    loss = params["Loss function"]
    try:
        regenerate_steps = params["Regeneration steps"]
    except KeyError:
        regenerate_steps = None
    

    conf_matrix = Metrics(classes_nr, cs_n=5 if classes_nr > 50 else 1, class_names=class_names, mode=mode)

    for ep in range(epochs):

        log.write("Epoch {}/{}".format(ep+1, epochs), toprint=True)

        train(model, train_dl, optimizer, loss, conf_matrix)
        log.write("TRAIN |  acc:  {} | MAE:  {} | CS[{x[0]}]: {x[1]}".format(conf_matrix.acc(), conf_matrix.mae(), x=conf_matrix.CS()), toprint=True, plain=True)
        if tblog != None:
            tblog.add_scalar('TRAIN/acc', conf_matrix.acc(), ep)
            tblog.add_scalar('TRAIN/mae', conf_matrix.mae(), ep)
            tblog.add_scalar('TRAIN/CS[{}]'.format(conf_matrix.CS()[0]), conf_matrix.CS()[1], ep)

        evaluate(model, valid_dl, conf_matrix)
        log.write("VALIDATION | acc: {} | MAE: {} | CS[{x[0]}]: {x[1]}".format(conf_matrix.acc(), conf_matrix.mae(), x=conf_matrix.CS()), toprint=True, plain=True)
        if tblog != None:
            tblog.add_scalar('VALID/acc', conf_matrix.acc(), ep)
            tblog.add_scalar('VALID/mae', conf_matrix.mae(), ep)
            tblog.add_scalar('VALID/CS[{}]'.format(conf_matrix.CS()[0]), conf_matrix.CS()[1], ep)
            tblog.add_figure('Validation Confusion Matrix for epoch {}'.format(ep+1), conf_matrix.plot())
            conf_matrix.plot().savefig(log.images_dir + "{}.png".format(ep+1))

        if scheduler is not None:
            if type(scheduler) == list:
                for s in scheduler: s.step()
            else: scheduler.step()
        
        if tblog != None:
            tblog.flush()

        if regenerate_steps != None:
            if len(regenerate_steps) > 0:
                if regenerate_steps[-1] == ep:
                    regenerate_steps.pop()
                    _, _, train_dl, valid_dl = regenerate_datasets()
            

run(NET, params, __mode__, train_loader, valid_loader, SCHED, log, tblog, class_names=classes_names) 

# gets the only file in tblog directory and saves in log directory
tblog_file = next(os.walk(tblog_name))[2][-1]
shutil.copyfile(tblog_name + '/' + tblog_file, logger.log_dir + '/tfevents' + log.name())

# NOTE: models are saved in a way that can be handled with Pytorch 1.4 (changed in 1.5 and could not update)
torch.save(NET.state_dict(), log.log_dir + "/train_{}.pth".format(os.path.basename(dataset_dir)), _use_new_zipfile_serialization=False)



