# NAGINI-3D | Prediction of Parametric Surfaces for Multi-Object Segmentation in 3D Biological Imaging

We present NAGINI-3D (N-Active shapes for seGmentINg 3D biological Images), a method dedicated to 3D biological images segmentation, based on both deep learning (CNN) and Active Surfaces (Snakes).

<img src="https://github.com/QuentinRapilly/NAGINI-3D/blob/main/images/logo/nagini.png" title="NAGINI Logo" width="25%" align="right">

This repository provides the code described in the paper:

- Quentin RAPILLY, Anaïs BADOUAL,Pierre MAINDRON, Guenaelle BOUET, Charles KERVRANN.
*Prediction of Parametric Surfaces for Multi-Object Segmentation in 3D Biological Imaging*.
Scale Space and Variational Methods in Computer Vision. SSVM 2025. Lecture Notes in Computer Science, vol 15667, Devon, UK, May 2025,
[(preprint)](https://hal.science/hal-04978619), [(final paper)](https://link.springer.com/chapter/10.1007/978-3-031-92366-1_20).

## Updates

**Version 0.1.1:** the method can now be trained more easily on highly anisotropic images.

## Method description

Our approach consists in training a U-Net to:

1. locate the objects of interest in a 3D image using a predicted probability map $\hat{p}$,
2. for each object, predict a set of control points $\lbrace\hat{{f}}\_{{x},i}\rbrace\_i$ describing a parametric surface $\hat{{s}}\_{{x}}$ representing the object located in ${x}$,
3. (optionnal) a snake optimisation procedure based on image gradient can be used to optimize the surfaces.

To evaluate the loss used to train the network, the Ground-Truth (GT) probability/spots map $p$ and a sampling $S$ representing each object of the training dataset are required. Some tools available in this Github will help you pre-process your data to create them.

The training and inference pipelines are summerized on the following figures.

**Training pipeline:**
![image](images/pipeline/training.png)

**Inference pipeline**
![image](images/pipeline/inference.png)

More details on the method are provided in the paper mentionned above.

## Installation

The experiments were run using Python 3.10.8. A list of all the packages installed to run the method is provided in [requirements.txt](requirements.txt).

### Installation using Singularity

To guarantee the proper functionning of the code and the reproducibility of the experiments, we recommend to create a Singularity container ([installation guide](https://docs.sylabs.io/guides/2.6/user-guide/installation.html)) using the same recipe as us. The recipe [nagini3D.def](singularity/nagini3D.def) and the corresponding requirement file [nagini3D.txt](singularity/nagini3D.txt) are provided in the [singularity directory](singularity).

Once Singularity is installed, run the following command to create the Singularity image:

`singularity build nagini3D.sif <path to nagini3D.def file>`

The image can then be run:

`singularity shell --nv -B <storage to your repository where the code and data are stored>:<storage to your repository where the code and data are stored> <path to the .sif image file>`

If the Docker image selected to create the Singularity image (see in [nagini3D.def](singularity/nagini3D.def)) doesn't match your GPU and CUDA compatibility, find another one [here](https://hub.docker.com/r/pytorch/pytorch/tags) that matches your requirements and with PyTorch>=1.13.

While the image is running, you should have the exact same version of Python, PyTorch and the important packages used to run the code.

### Installation using pip

Our implementation can be installed using pip.

We strongly recommend installing it in a dedicated conda environment.

`conda create --name nagini-env python=3.10.8`

`conda activate nagini-env`

Before installing our package, the only requirement is that your environment has a working version of Pytorch>=1.13.
If you want to use your GPU (strongly recommended, especially for training), make sure you have a working CUDA.

No GPU: `conda install pytorch=1.13`

Using GPU: `conda install pytorch=1.13 pytorch-cuda=11.6 -c pytorch -c nvidia` (you might need to change the CUDA version depending on your GPU requirements and driver).

To make sure your GPU is available, open a Python prompt and run the following commands:

```python
from torch.cuda import is_available
cuda_working = "yes" if is_available() else "no"
print(f"GPU working: {cuda_working}")
```

If it outputs "no", something didn't work in your CUDA installation.

To install the package, run:

`pip install nagini3D`

## Using the method

All the scripts are designed to process TIF images.

### Learning how to use the package using the jupyter notebooks

We provide jupyter notebooks (in the [tutorials directory](/tutorials/)) to teach you how to infere new data, pre-process your data for training, train a model.

- [Inference notebook](/tutorials/inference.ipynb)
- [Formating data notebook](/tutorials/format_data.ipynb)
- [Training](/tutorials/training.ipynb)

### Applying our scrpits directly

We provide all the scripts we used to make our experiments. To use it, we recommend installing an extended version of your package that contains additional packages:

- **wandb** (to display training logs on wandb.ai, it requires creating an account),
- **hydra** (used for configuration files management),
- **tifffile** (used to load the tiff images).

To install it run: `pip install nagini3D[full]` instead of the classical pip command (see "Installation using pip"). If you choose the Singularity installation, everything is already installed.

#### Preprocessing data for training

The script [format_dataset.py](format_dataset.py) pre-processes the GT masks to create the probability maps and the sampling of the objects.

`python format_dataset.py -i <str: directory containing the masks> -o <str: directory to store the samplings and spot maps> -n <int (optionnal, default 101): number of points to sample on the surface> -v <bool: verbose>`

Warnings:

1. Here, the sampling procedure can produce any positive integer number of points. But the sampling procedure used for predicted surfaces (Fibonacci lattice) during training requires an odd number of sampled points. Make sure that the sampling size is greater or equal than the sampling size you will use during training.
2. Make sur that your labels are indexed contiguously (no missing labels, ex: 1,2,4 but no mask correspond to index 3).

#### Formating dataset for training

The repository containing each dataset (training, test, validation) should be organized as follow:

```txt
directory_of_the_set
|--images     (directory containing the images of the set)
|--masks      (directory containing the masks, with the same name as the corresponding image)
|--samplings  (directory containing the output of the "format_dataset.py" script)
```

#### Training a model

Edit the file [configs/train.yaml](configs/train.yaml), then launch the scrpit [train.py](train.py).

`python train.py`

If the wandb option is activated, you can follow the train logs on [wandb.ai](https://wandb.ai).

#### Infering on new data

Run the file [inference_on_dir.py](inference_on_dir.py):

`python inference_on_dir.py -i <images directory> -o <directory to store the results> -m <directory containing the trained model and its config file> -s <bool: weither to apply a snake optimisation step after the network prediction>`

Optionnal parameters:

- `-t <(float,float): probability threshold used to extract local maxima and NMS thresholds used to remove duplicates>`. If the training finished correctly, the last step consists in evaluating the best thresholds on the validation set, in this case, you don't need to provide this parameter.
- `-tt <(int,int,int): number of tiles to do along each dimension>`. By default set to (1,1,1), can be useful to split some images in tiles if they are too big for your GPU/CPU.
- `-ot <bool: if True, apply an Otsu binarization of the image before snake optimization>`. For sparse objects, this option improves drastically the results. For dense objects, keep it to False.

## Dataset

To test the algorithm, we provide the CAPS dataset described in the article (the weights of the network obtained after being trained on it are provided in the next sectio).

[Link to CAPS dataset.](https://zenodo.org/records/14931808?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjlhYTBlMTRiLTI3YWMtNGIxNi1iNTQxLTcwNjFjMjFlYmE3YiIsImRhdGEiOnt9LCJyYW5kb20iOiI2MDdmMmQ3NzdjZWMyNDM1NTA4ZjI4OTUzYmQ3OWU3MiJ9.rJK8i6DmDl75V3fxJNIm63LeXsm0uHrOGoOc4mtiYOxBLGSAzfzfu04QlZft5eKr38c-r8exYpDE_ZqqBURldg)

## Pre-trained weights

[Link to the network config file and weights after being trained on CAPS (+optimal thresholds used for inference).](https://zenodo.org/records/14932135?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjQ5YjY5Mzc0LWRhOWUtNGExZi05YmQ4LTMxOWI1ZWFjYTFiMCIsImRhdGEiOnt9LCJyYW5kb20iOiIzMTNhODA1ZjEzZTYwZDRjNWRhMjMzYzk4MDkxYTIwYyJ9.m8pDDXwVZarpL_sEgtrvMztJgMBaQa_VkusZTIROr-BqkyUI8WNp7MqQI22Si1OfxWNIhp8ei6SCVJFI83iWJg)

## How to cite this method

```bibtex
@InProceedings{10.1007/978-3-031-92366-1_20,
author="Rapilly, Quentin and Badoual, Ana{\"i}s and Maindron, Pierre and Bouet, Guenaelle and Kervrann, Charles",
editor="Bubba, Tatiana A. and Gaburro, Romina and Gazzola, Silvia and Papafitsoros, Kostas and Pereyra, Marcelo and Sch{\"o}nlieb, Carola-Bibiane",
title="Prediction of Parametric Surfaces for Multi-object Segmentation in 3D Biological Imaging",
booktitle="Scale Space and Variational Methods in Computer Vision",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="255--268",
isbn="978-3-031-92366-1"
}

```
