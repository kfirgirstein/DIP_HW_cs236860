import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import fftpack

def plot_image(image_, name_: str,path_prefix=None):
    plt.figure()
    plt.imshow(image_, cmap="gray")
    plt.title(name_)
    plt.show(block=0)
    if path_prefix:
        plt.savefig(path_prefix + name_)
    return

def plot_psf(psf,name):
    plt.imshow(psf, cmap='gray')
    plt.title(f'{name} PSF (spatial domain)')
    plt.show()
    psf_dft = fftpack.fftshift(fftpack.fft2(psf))
    plt.title(f'{name} PSF (frequency domain)')
    plt.imshow(np.abs(psf_dft))
    plt.colorbar()
    plt.show()
