# AgeRecog
A deep-learning based computer vision project on estimating people's age from face images. The aim of this project was to see if age regression could
be performed also with compact networks, in order to make architectures deployable in real-time scenarios. 

This project was the main part of a dissertation for a bachelor's degree in Computer Science at the University of Trento in September 2020. It was developed during an internship at Dept. of Information Engineering and Computer Science, that lasted
from June 2020 to September 2020.

## Usage information

Dependencies can be installed from the `requirements.txt` file using `pip` or `conda`.

In order to execute the **real-time demo**, install dependencies and launch the command:
```
python realtime_demo.py
```


### Expected dataset structure

In order to replicate the experiments, the following structure must be replicated:

```
<current_dir>
  |___dataset_parent_dir
	|____dataset_compressed_archive (optional)
	|____dataset_dir
	|	|___age1
	|	|___age2
	|	|___...
	|	|___agemax
	|____logs
		|___day1
		|    |___exp1
		|    |___exp2
		|___day2
		|___...
```

### General Notes
- datasets are expected to be either in a .tar.gz file (eventually more extensions can be added) or in a directory without subdirectories;
  if some directory is found in `dataset` folder, the script will interpret it as already divided in subdirectories for each age class;

- if there is no subdirectory in dataset folder, then the training script automatically sorts the dataset by retrieving age from images' 
  path and placing them in the correct age-directory, in order to use `torchvision.datasets.ImageFolder`;

- the method to extract age from path must be adapted to the dataset in use;

- logs are stored in the dataset parent directory, divided by day and by time of execution;

- `train.py` expects to be given training parameters (dataset paths, learning rate, batch sizes...) by modifying them inside the script: most of them are inside a `params` dictionary, but not all, and dataset path is at the beginning of the script;

- models are supposed to be saved and loaded as was done in PyTorch 1.4, thus NOT using the new zipfile serialization introduced in PyTorch 1.5;

- `retrain.py` accepts parameters from command line, but just the ones to define model architecture (not training parameters);

- retrained models are saved in a special directory called `retrain_models`, created by retrain script, that has same structure of `logs`

- `plot.py` is used to plot experiments comparisons, but paths and names of experiments must be coded inside the script at each use;

- `dataset.py` presents an incomplete method to get datasets in an easiest way than hard-coding in scripts, but it is not actually used, while `retrain.py` presents a dictionary to get them from command line;

- `refactor_imdb.py` is a tool to produce the correct directory structure on IMDB-WIKI dataset: it has been tested to work, but in the end this dataset has not been implemented in final work;

- `realtime_demo.py` accepts images, videos and real-time webcam when no file is specified; results are stored in 'demo_' directories, with a folder for each different model tested (applying same model to same file will result in deleting the previous version of results).
