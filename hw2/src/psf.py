import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import fftpack,signal
import scipy.signal
import skimage.measure

class PSFManager:
        
    def CreatePSF(self,name:str, **kwargs):
        choice = name.lower()
        if choice == 'sinc':
            return self.__sinc__(**kwargs)
        elif choice == 'gaussian':
            return self.__gaussian_kernel__(**kwargs)
        elif choice == 'box':
            return self.__box__(**kwargs)
        else: 
            raise Exception(f"invalid name {name}")
    
    def ApplyPSF(self,image,psf):
        if psf is None or image is None:
             raise Exception(f"invalid parames")
        psf_image = signal.convolve2d(image, psf, mode='same', boundary='wrap')
        psnr = skimage.metrics.peak_signal_noise_ratio(image, psf_image)
        return (psf_image,psnr)
    
        
    def __sinc__(self,window_size):
        edge = window_size // 2
        x = np.linspace(-edge, edge, num=window_size)
        xx = np.outer(x, x)
        s = np.sinc(xx)
        s = s / s.sum()
        return s

    def __box__(self,width,height,box_size):
        h_ = int(height / 2)
        w_ = int(width / 2)
        d_ = int(box_size / 2)
        PSF_box_ = np.zeros((height, width))
        PSF_box_[h_ - d_:h_ + d_, w_ - d_:w_ + d_] = 1 / (box_size ** 2)
        return PSF_box_

    def __gaussian_kernel__(self,size, std=1):
        edge = size // 2
        ax = np.linspace(-edge, edge, num=size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * std **2))
        return kernel / kernel.sum()
