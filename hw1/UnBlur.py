import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import signal
from scipy import fftpack
import scipy.io


class Blurr_Fixer:
    def __init__(self, blurred_images, p=1,ifft_scale=1000, original_size=256, margin=0):
        self.blurred_images = blurred_images
        self.F_images = list()
        self.p = p
        self.ifft_scale = ifft_scale
        for img in blurred_images:
            self.F_images.append(fftpack.fftshift(fftpack.fftn(img)))
        self.fixed = []
        self.margin = margin
        self.original_size = original_size

    def calc_weights_denom(self):
        weights_denom = np.zeros(self.F_images[0].shape)
        for mat in self.F_images:
            weights_denom = weights_denom + np.power(np.abs(mat), self.p)
        return weights_denom

    def fix_blurr(self):
        denom = self.calc_weights_denom()
        accumulator = np.zeros(self.F_images[0].shape)
        for F in self.F_images:
            curr_weight = np.divide(np.power(np.abs(F), self.p), denom)
            accumulator = accumulator + np.multiply(F, curr_weight)
        fixed = fftpack.ifft2(fftpack.ifftshift(accumulator)).real
        fixed = np.divide(fixed,self.ifft_scale)
        # Crop
        size = self.original_size
        margin = self.margin
        self.fixed = fixed[margin:margin + size, margin:margin + size]
        return self.fixed

    def get_FT(self):
        return self.F_image

    def show_Fixed(self,path):
        original = mpimg.imread(path)
        p1 = plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray')
        p1.set_title("Original Image")
        p2 = plt.subplot(1, 2, 2)
        plt.imshow(self.fixed, cmap='gray')
        p2.set_title("Fourier Burst Accumulation")
        plt.show()
