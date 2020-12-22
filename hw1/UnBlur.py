import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import signal
from scipy import fftpack
import scipy.io


class Blurr_Fixer:
    def __init__(self, blurred_images, power=1,ifft_scale=1000, original_size=256, margin=0):
        self.blurred_images = blurred_images
        self.power = power
        self.ifft_scale = ifft_scale
        self.fixed = []
        self.margin = margin
        self.original_size = original_size
        self.F_images = [fftpack.fftshift(fftpack.fftn(img)) for img in blurred_images]

    def __get_weights_denom(self):
        weights_denom = np.zeros(self.F_images[0].shape)
        for mat in self.F_images:
            weights_denom = weights_denom + np.power(np.abs(mat), self.power)
        return weights_denom

    def unblur_images(self):
        denom = self.__get_weights_denom()
        accumulator = np.zeros(self.F_images[0].shape)
        for F in self.F_images:
            curr_weight = np.divide(np.power(np.abs(F), self.power), denom)
            accumulator = accumulator + np.multiply(F, curr_weight)
        fixed = fftpack.ifft2(fftpack.ifftshift(accumulator)).real
        fixed = np.divide(fixed,self.ifft_scale)
        # Crop
        size = self.original_size
        margin = self.margin
        self.fixed = fixed[margin:margin + size, margin:margin + size]
        return self.fixed

    def show_unblur_image(self,original_image):
        p1 = plt.subplot(1, 2, 1)
        plt.imshow(original_image.get_image(), cmap='gray')
        p1.set_title("Original")
        p2 = plt.subplot(1, 2, 2)
        plt.imshow(self.fixed, cmap='gray')
        p2.set_title("Fourier Burst Accumulation")
        plt.show()
