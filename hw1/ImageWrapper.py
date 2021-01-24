import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import signal
from scipy import fftpack
import scipy.io

class ImageWrapper:
    def __init__(self,path):
        self.image = mpimg.imread(path)
        self.image_arr = np.array(self.image)[:, :, 0]

    def get_2d_array(self):
        return self.image_arr

    def get_image(self):
        return self.image

    def apply_filter(self,image_filter,show = False):
        filtered = scipy.signal.convolve2d(self.image_arr,image_filter,mode = 'same', boundary = 'wrap')

        if show == True:
            plt.imshow(filtered, cmap='gray')
            plt.show()
        return filtered