r"""
Use this module to write hyper parameters in the notebook.
"""
window_size = 15

def sinc_hp():
    return dict(window_size=256,filter_range=window_size//4)


def gaussian_hp():
    filter_range,mu,sigma = window_size//16,0,1
    return dict(window_size=window_size,filter_range=filter_range,mu=mu, sigma=sigma)

def esimator_hp():
    iterations,patch_size,reduced_patch_size,num_neighbors,sigma = 5,5,10,10,0.3
    return dict(iterations=5,patch_size=patch_size,reduced_patch_size=reduced_patch_size,num_neighbors=num_neighbors,sigma=sigma)