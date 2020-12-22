import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import signal
from scipy import fftpack
import scipy.io

class Blur:
    def __init__(self,trajectories,image):
        self.trajectories = trajectories
        self.__generate_PSFs()
        self.__apply_blur(image)

    def get_blurred_images(self):
        return self.blurred_images
    
    def __generate_PSFs(self):
        self.psfs = list()
        for i in range(len(self.trajectories)):
            self.psfs.append(self.trajectories.generate_kernel(i))

    def __apply_blur(self,image):
        self.blurred_images = list()
        for i in range(len(self.trajectories)):
            self.blurred_images.append(image.apply_filter(self.psfs[i]))

    def plot_blured_batch(self,batch):
        for i in range(batch):
            x,y = self.trajectories.get_trajectory(i)
            plt.subplot(3, 5, i+1)
            plt.plot(x,y)
            plt.subplot(3, 5, batch+i+1)
            plt.imshow(self.psfs[i], cmap='gray')
            plt.subplot(3, 5, 2*batch+i+1)
            plt.imshow(self.blurred_images[i], cmap='gray')
        plt.show()
            
    def plot_blured_batches(self,batch_size):
        for batch in range(0,20,1):
            n = batch_size
            for t in range(5):
                x,y = self.trajectories.get_trajectory(t+batch*5)
                plt.subplot(3, 5, t+1)
                plt.plot(x,y)
                plt.subplot(3, 5, n+t+1)
                plt.imshow(self.psfs[t+batch*5], cmap='gray')
                plt.subplot(3, 5, 2*n+t+1)
                plt.imshow(self.blurred_images[t+batch*5], cmap='gray')
            plt.show()

    def save_plot_to_image(self,path):
        for (x,y) in self.trajectories:
            plt.subplot2grid((2, 3), (0, 2))
            plt.plot(x,y)
            plt.subplot2grid((2, 3), (1, 2))
            plt.imshow(self.psfs[t], cmap='gray')
            plt.subplot2grid((2, 3), (0, 0),colspan=2, rowspan=2)
            plt.imshow(self.blurred_images[t], cmap='gray')
            plt.tight_layout()
            full_path = path + str(t) + '.png'
            plt.savefig(full_path)
            plt.show()
            plt.clf()

