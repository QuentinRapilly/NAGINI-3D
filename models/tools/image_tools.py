import numpy as np

def trilinear(xyz, data):
    '''
    xyz: array with coordinates inside data
    data: 3d volume
    returns: interpolated data values at coordinates
    '''
    nx, ny, nz = data.shape
    ijk = xyz.numpy().astype(np.int32)
    i, j, k = ijk[...,0], ijk[...,1], ijk[...,2]
    i = (i>0)*(i<(nx-1))*i + (nx-2)*(i>=(nx-1))
    j = (j>0)*(j<(ny-1))*j + (ny-2)*(j>=(ny-1))
    k = (k>0)*(k<(nz-1))*k + (nz-2)*(k>=(nz-1))
    V000 = data[ i   , j   ,  k   ]
    V100 = data[(i+1), j   ,  k   ]
    V010 = data[ i   ,(j+1),  k   ]
    V001 = data[ i   , j   , (k+1)]
    V101 = data[(i+1), j   , (k+1)]
    V011 = data[ i   ,(j+1), (k+1)]
    V110 = data[(i+1),(j+1),  k   ]
    V111 = data[(i+1),(j+1), (k+1)]
    xyz = (xyz - ijk).numpy()
    x, y, z = xyz[...,0], xyz[...,1], xyz[...,2]
    Vxyz = (V000 * (1 - x)*(1 - y)*(1 - z)
            + V100 * x * (1 - y) * (1 - z) +
            + V010 * (1 - x) * y * (1 - z) +
            + V001 * (1 - x) * (1 - y) * z +
            + V101 * x * (1 - y) * z +
            + V011 * (1 - x) * y * z +
            + V110 * x * y * (1 - z) +
            + V111 * x * y * z)
    return Vxyz