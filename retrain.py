import os, random, tarfile, argparse
import shutil

import torch
import torchvision
from torch.utils import tensorboard as tb

import dataset
from utils import *
from loss import *
from net import get_model


parser = argparse.ArgumentParser()

# type of model to retrain, can be '2classifiermn', 'mobilenet' or 'resnet18'
parser.add_argument('model', type=str)

parser.add_argument('num_classes', type=int)
parser.add_argument('model_path', type=str)
parser.add_argument('from_dataset', type=str)
parser.add_argument('to_dataset', type=str)

# Possibility to implement 3channel grayscale (also 1channel with 2classifiermn), but never used
parser.add_argument('--grayscale', type=int, choices=[1, 3])

parser.add_argument('--freeze_features', action='store_true')

datasets = {
    'fgnet': ("FGNET", "D:\\projects_coding\\internship\\FGNet_dataset/"),
    'utkface': ("UTKFace", "UTK_face_dataset/"),
    'imdb': ("imdb_refined", "imdb/")
}

################## Dataset choice ##################

args = parser.parse_args()
    
dataset_dir, drive_dataset_dir = datasets[args.to_dataset]

dataset_dir = drive_dataset_dir + dataset_dir

__nr_total_classes__ = -1
__nr_total_images__ = 0
_, __total_classes__, _ = next(os.walk(dataset_dir))
if __nr_total_images__ < 0:
    __nr_total_images__ = 0
    for _, _, files in os.walk(dataset_dir):
        __nr_total_images__ += len(files)

__nr_total_classes__ = max([int(el) for el in __total_classes__]) + 1 


################## Parameters and initialization ##################

params = {
    # used to evidentiate first line of retrain log file
    "Retrain": "###############################",
    "Classes number": args.num_classes,
    "Image size": (224, 224),
    "Dataset sizes": {'train': 900, 'valid': 100, 'test': 50},
    "Batch size": 32,
    "Epochs": 20,
    "Learning rate": 1e-4,
    "Loss function": MVLoss(),
    "Grayscale": args.grayscale
}


complete_dataset = torchvision.datasets.ImageFolder(dataset_dir, transform=None, target_transform=None)

# needed to correctly retrieve age label
inv_dict = {v: int(k) for k, v in complete_dataset.class_to_idx.items()}

# used to extract age label when loading batches
target_transform = (lambda t: inv_dict[t])
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
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.Resize(params["Image size"]),
        torchvision.transforms.ToTensor()
    ])

    valid_transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=params["Grayscale"]) if params["Grayscale"] != None else identity_transf,
        torchvision.transforms.Resize(params["Image size"]),
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
    
    # NOTE: the last returned value should be test data indices, but in reality are never used
    return train_dataset, valid_dataset, train_loader, valid_loader, indices[2]


train_dataset, valid_dataset, train_loader, valid_loader, indices_test = regenerate_datasets(complete_dataset)


model_file_name = os.path.join(args.model_path, get_model_fname(args.model_path))
model = get_model(args.model, num_classes=args.num_classes).cuda()
model.load_state_dict(torch.load(model_file_name))


params["Model"] = model._get_name()


log = logger(params, topdir='./retrain_models')

if args.freeze_features:
    for layer in model.features:
        for param in layer.parameters():
            param.requires_grad = False
    log.write("Freezing convolutional layers", plain=True)



params["Optimizer"] = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params["Learning rate"])
scheduler = None
scheduler = torch.optim.lr_scheduler.ExponentialLR(params["Optimizer"], 0.92)



# saves the entire data in a tensorboard file (can be viewed with tensorboard dev)
# first deletes the content of './tblog' directory to avoid accumulation of multiple runs
if os.path.exists('./tblog'):
    shutil.rmtree('./tblog')
    os.mkdir('./tblog')

tblog = None
tblog_name = './tblog'; tblog = tb.SummaryWriter(tblog_name);


################## Training ##################

def predict(model, x):
    """ Predicts values as mean of the output distribution, when using MV-Loss as error function."""
    yp = model(x).cpu()
    # the actual prediction is the mean value of the distribution, not the max!
    prediction = torch.sum(
        torch.nn.functional.softmax(yp,dim=1) * torch.arange(args.num_classes, device=yp.device), 
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
            yt = yt.long()

        if isinstance(loss, MVLoss):
            # if using different approach then must be a probability
            yp, prediction = predict(model, X)
        else: 
            # else it is simply the maximum
            yp = model(X)
            yp = yp.cpu()
            prediction = yp.argmax(1)
            
    
        if isinstance(loss, MVLoss):
                sfloss, mean_loss, var_loss = loss(yp, yt)
                err = sfloss + mean_loss + var_loss
        else:
            sfloss, mean_loss, var_loss = loss(yp, yt), None, None
            err = sfloss

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

            metrics.add(prediction, yt)


    


def run(model, params, mode, 
        train_dl, valid_dl, scheduler, 
        log, tblog=None, class_names=None):
    
    classes_nr = params["Classes number"]
    epochs = params["Epochs"]
    optimizer = params["Optimizer"]
    loss = params["Loss function"]

    # eventually gets epochs at which datasets shall be regenerated (not used in final thesis)
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
            scheduler.step()
        
        if tblog != None:
            tblog.flush()

        if regenerate_steps != None:
            if len(regenerate_steps) > 0:
                if regenerate_steps[-1] == ep:
                    regenerate_steps.pop()
                    _, _, train_dl, valid_dl, _ = regenerate_datasets()
            

run(model, params, 'regression', train_loader, valid_loader, scheduler,  log, tblog, class_names=None) 


# saves tblog file in log directory   
tblog_file = next(os.walk(tblog_name))[2][-1] 
shutil.copyfile(tblog_name + '/' + tblog_file, logger.log_dir + '/tfevents' + log.name())

# NOTE: models are saved in a way that can be handled with Pytorch 1.4 (changed in 1.5 and could not update)
torch.save(model.state_dict(), log.log_dir + "/retr_from_{}_to_{}.pth".format(args.from_dataset, args.to_dataset), _use_new_zipfile_serialization=False) 
