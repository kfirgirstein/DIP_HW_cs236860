import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import signal
from scipy import fftpack
import scipy.io

class PSNR_calculator:
    def __init__(self,original_image,fixed_image):
        self.orig = original_image[:,:,0]
        self.fixed = np.array(fixed_image)

    def evaluate_PSNR(self):
        MSE = np.mean(np.power(np.abs(np.subtract(self.fixed,self.orig)),2))
        MAX = 255
        return 20*np.log10(MAX)-10*np.log10(MSE)

