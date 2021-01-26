import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import fftpack,signal
import scipy.signal
import skimage.measure

class DeblurFactory:
        
    def ApplyDeblurFilter(self,name:str, **kwargs):
        choice = name.lower()
        if choice == 'wiener':
            return self.__wiener_filter__(**kwargs)
        else: 
            raise Exception(f"invalid name {name}")
    
    def evaluate(self,restored,target):
        return skimage.metrics.peak_signal_noise_ratio(restored, target)
    
    def __wiener_filter__(image, psf, k=0.01):
        image_dft = fftpack.fft2(image)
        psf_dft = fftpack.fft2(fftpack.ifftshift(psf))
        filter_dft = np.conj(psf_dft) / (np.abs(psf_dft) ** 2 + k)
        recovered_dft = image_dft * filter_dft
        return np.real(fftpack.ifft2(recovered_dft))
