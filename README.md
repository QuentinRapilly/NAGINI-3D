# NAGINI-3D | Prediction of Parametric Surfaces for Multi-Object Segmentation in 3D Biological Imaging

We present NAGINI-3D (N-Active shapes for seGmentINg 3D biological Images), a method dedicated to 3D biological images segmentation, based on both deep learning (CNN) and Active Surfaces (Snakes).

<img src="https://github.com/QuentinRapilly/NAGINI-3D/blob/main/images/logo/nagini.png" title="NAGINI Logo" width="25%" align="right">

This repository provides the code described in the paper:
- Quentin RAPILLY, Pierre MAINDRON, Anaïs BADOUAL, Guenaelle BOUET-CHALON, Charles KERVRANN.
[*Prediction of Parametric Surfaces for Multi-Object Segmentation in 3D Biological Imaging*](TODO_ajouter_le_lien).
International Conference on Scale Space and Variational Methods in Computer Vision (SSVM), Totnes, England, May 2025.

## Method description

Our approach consists in training a U-Net to:
1. locate the objects of interest in an 3D image using a probability map,
2. for each object, predict a set of control points describing a parametric surface representing the object,
3. (optionnal) a snake optimisation procedure based on image gradient can be used to optimize the surfaces.

To evaluate the loss used to train the network, the probability map and a sampling representing each object of the training dataset are required. Some tools available in this Github will help you pre-process you data to create them.

![image](https://github.com/QuentinRapilly/NAGINI-3D/blob/main/logo/nagini.png)