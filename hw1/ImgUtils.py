import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import signal
from scipy import fftpack
import scipy.io

class ImageUtils():
    def __init__(self, blurred_images):
        self.blurred_images = blurred_images
        self.num_img = len(blurred_images)

    def get_avg(self):
        avg_img = np.zeros(self.blurred_images[0].shape)
        for img in self.blurred_images:
            avg_img = avg_img + np.divide(img,self.num_img)
        return avg_img

    def get_sharp_avg(self):
        avg = self.get_avg()
        kernel = [[0,-2,0],[-2,9,-2],[0,-2,0]]
        sharpen = scipy.signal.convolve2d(avg, kernel)
        return sharpen

