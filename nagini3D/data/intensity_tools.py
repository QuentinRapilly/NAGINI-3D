import numpy as np
from random import random, randint

def alter_image(image, alteration, verbose = False):

    if verbose: print(f"Alteration : {alteration['name']}")

    if alteration["name"] == "none":
        return image

    if alteration["name"] == "noise":
        img_shape = image.shape
        noise = np.random.normal(0, alteration["noise_lvl"], img_shape)
        new_img = image + noise
        return new_img
    
    if alteration["name"] == "gamma":
        return image**alteration["gamma"]
    
    if alteration["name"] == "varying_shift":
        nx, ny, nz = image.shape
        vx, vy, vz = np.arange(nx)[:, None, None], np.arange(ny)[None,:,None], np.arange(nz)[None,None,:]
        rx, ry, rz = random(),random(),random()
        vx = vx if rx<0.34 else (nx-vx-1 if rx > 0.67 else 0)
        vy = vy if ry<0.34 else (ny-vy-1 if ry > 0.67 else 0)
        vz = vz if rz<0.34 else (nz-vz-1 if rz > 0.67 else 0)
        mu = alteration["mu"]/3
        return image + mu*(vx/nx + vy/ny + vz/nz) 
    
    if alteration["name"] == "piecewise_shift":
        mu = alteration["mu"]
        nx, ny, nz = image.shape
        vx, vy, vz = np.arange(nx)[:, None, None], np.arange(ny)[None,:,None], np.arange(nz)[None,None,:]
        multx, multy, multz = 2*randint(0,1)-1, 2*randint(0,1)-1, 2*randint(0,1)-1
        splitx, splity, splitz = randint(1, nx-2), randint(1, ny-2), randint(1, nz-2)
        return image + mu*(multx*vx<=multx*splitx)*(multy*vy<=multy*splity)*(multz*vz<=multz*splitz)

    
    else:
        print("Unknown alteration")
        assert False

MAX_NOISE = 0.08
MAX_SHIFT = 0.4
MIN_SHIFT = - MAX_SHIFT
MAX_SCALE = 4
MAX_GAMMA = 5
INV_GAMMA = 1/MAX_GAMMA
FIXED_NOISE = 0.05




def generate_alteration_parameters(alteration_name):

    if alteration_name == "none":
        alteration = {
            "name": "none"
        }

    if (alteration_name == "varying_shift") or (alteration_name=="piecewise_shift"):
        alteration = {
            "name": alteration_name,
            "mu" : 0.1 + random()*MAX_SHIFT
        }

    if alteration_name == "noise":
        alteration = {
            "name": alteration_name,
            "noise_lvl": MAX_NOISE*random()
            }
    
    if alteration_name == "noise_fixed":
        alteration = {
            "name": "noise",
            "noise_lvl": FIXED_NOISE
        }
        
    if alteration_name == "gamma":
        bright = random()>0.5
        x = INV_GAMMA+(1-INV_GAMMA)*random()
        alteration = {
            "name": alteration_name,
            "gamma": x if bright else 1/x
        }

    
    if alteration_name == "affine":
        alteration = {
            "name": "affine",
            "scale":  (random() if random()>0.5 else 1+random()*(MAX_SCALE-1)),
            "shift":  MIN_SHIFT + (MAX_SHIFT - MIN_SHIFT)*random(),
            "saturated": False
        }


    if alteration_name == "affine_saturated":
        alteration = {
            "name": "affine",
            "scale":  0.5 + random(),
            "shift": (1-2*(random()>0.5))*(0.4+0.4*random()),
            "saturated": True
        }

    if alteration_name == "noise_high":
        alteration = {
            "name": "noise",
            "noise_lvl": 0.15 + 0.1*random()
        }

    if alteration_name == "noise_low":
        alteration = {
            "name": "noise",
            "noise_lvl": 0.01 + 0.02*random()
        }

    if alteration_name == "shift":
        alteration = {
            "name": "affine",
            "scale": 1,
            "shift": MIN_SHIFT + (MAX_SHIFT-MIN_SHIFT)*random(),
            "saturated": False
        }

    if alteration_name == "scale":
        alteration = {
            "name": "affine",
            "scale": (random() if random()>0.5 else 1+random()*(MAX_SCALE-1)),
            "shift": 0,
            "saturated": False
        }
    
    if alteration_name == "scale_low":
        alteration = {
            "name": "affine",
            "scale": random(),
            "shift": 0,
            "saturated": False
        }

    if alteration_name == "scale_high":
        alteration = {
            "name": "affine",
            "scale": 1 + (MAX_SCALE-1)*random(),
            "shift": 0,
            "saturated": False
        }

    if alteration_name == "gamma_dark":
        alteration = {
            "name": "gamma",
            "gamma": 1/(INV_GAMMA+(1-INV_GAMMA)*random())
        }

    if alteration_name == "gamma_light":
        alteration = {
            "name": "gamma",
            "gamma": (INV_GAMMA+(1-INV_GAMMA)*random())
        }

    if alteration_name == "shift_saturated":
        alteration = {
            "name": "affine",
            "scale": 1,
            "shift": (1-2*(random()>0.5))*(0.4+0.4*random()),
            "saturated": True
        }

    if alteration_name == "scale_saturated":
        alteration = {
            "name": "affine",
            "scale": 1+3*random(),
            "shift": 0,
            "saturated": True
        }
    
    
    return alteration
