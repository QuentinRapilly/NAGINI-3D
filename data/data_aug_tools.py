from math import cos, sin
import numpy as np
from skimage.filters import gaussian
from random import random

def generate_Rx(theta):
    return np.array([
        [1, 0, 0],
        [0, cos(theta), -sin(theta)],
        [0, sin(theta), cos(theta)]
    ])

def generate_Ry(theta):
    return np.array([
        [cos(theta), 0, sin(theta)],
        [0, 1, 0],
        [-sin(theta), 0, cos(theta)]
    
    ])

def generate_Rz(theta):
    return np.array([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0, 0, 1],
    ])

def generate_rotation_matrix(theta1, theta2, theta3):
    return generate_Rz(theta3)@generate_Ry(theta2)@generate_Rx(theta1)

def rotate_image_along_one_dim(img, dim1, dim2, angle):
    """
        angle should be in {0,1,2,3} corresponding to a real angle of angle*pi/2
    """
    if angle%2==1:
        img = np.swapaxes(img, axis1 = dim1, axis2 = dim2)
    if angle>0 and angle<3:
        img = np.flip(img, axis=dim1)
    if angle>1:
        img = np.flip(img, axis=dim2)
    return img

def rotate_image(img, angles):
    angle1, angle2, angle3 = angles
    img = rotate_image_along_one_dim(img, 1, 2, angle1)
    img = rotate_image_along_one_dim(img, 2, 0, angle2)
    return rotate_image_along_one_dim(img, 0, 1, angle3)

def blur_image(img, sigma_max):
    return gaussian(img, sigma_max*random(), preserve_range=True)