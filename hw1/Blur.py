import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import signal
from scipy import fftpack
import scipy.io

class Blur:
    def __init__(self,num,trajectories):
        self.num_of_traj = num
        self.trajectories = trajectories
        self.psfs = list()
        self.blurred_images = list()

    def generate_PSFs(self):
        for i in range(self.num_of_traj):
            self.psfs.append(self.trajectories.generate_kernel(i))

    def apply_blur(self,image):
        for i in range(self.num_of_traj):
            self.blurred_images.append(image.apply_filter(self.psfs[i]))

    def plot_blured_batch(self,batch):
            n = batch
            for t in range(5):
                x,y = self.trajectories.get_trajectory(t+batch*5)
                plt.subplot(3, 5, t+1)
                plt.plot(x,y)
                plt.subplot(3, 5, n+t+1)
                plt.imshow(self.psfs[t+batch*5], cmap='gray')
                plt.subplot(3, 5, 2*n+t+1)
                plt.imshow(self.blurred_images[t+batch*5], cmap='gray')
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

    def plot_batches_of_10(self):
        for batch in range(0,10,1):
            n = 10 # num of images in batch
            for t in range(10):
                x,y = self.trajectories.get_trajectory(t+batch*5)
                plt.subplot(3, 10, t+1)
                plt.plot(x,y)
                plt.subplot(3, 10, n+t+1)
                plt.imshow(self.psfs[t+batch*10], cmap='gray')
                plt.subplot(3, 10, 2*n+t+1)
                plt.imshow(self.blurred_images[t+batch*10], cmap='gray')
            plt.show()

    def save_plot_to_image(self,path):
            for t in range(self.num_of_traj):
                x,y = self.trajectories.get_trajectory(t)
                plt.subplot2grid((2, 3), (0, 2))
                plt.plot(x,y)
                plt.subplot2grid((2, 3), (1, 2))
                plt.imshow(self.psfs[t], cmap='gray')
                plt.subplot2grid((2, 3), (0, 0),colspan=2, rowspan=2)
                plt.imshow(self.blurred_images[t], cmap='gray')
                plt.tight_layout()
                full_path = path + str(t) + '.png'
                plt.savefig(full_path)
                #plt.show()
                plt.clf()


    def get_blurred_images(self):
        return self.blurred_images
