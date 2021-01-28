r"""
Use this module to write hyper parameters in the notebook.
"""
SIZE = 256

def sinc_hp():
    window_size = SIZE
    return dict(window_size=window_size)

def gaussian_hp():
    size,sigma =SIZE,1
    return dict(size=size,std=sigma)

def esimator_hp():
    iterations,patch_size,reduced_patch_size,num_neighbors,sigma = 5,5,10,10,0.3
    return dict(iterations=5,patch_size=patch_size,reduced_patch_size=reduced_patch_size,num_neighbors=num_neighbors,sigma=sigma)