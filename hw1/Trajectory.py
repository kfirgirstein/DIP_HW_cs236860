import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import signal
from scipy import fftpack
import scipy.io


class Trajectories:
    def __init__(self,path):
        mat = scipy.io.loadmat(path)
        if not mat:
            raise FileNotFoundError("")
        self.x_vals = mat['X']
        self.y_vals = mat['Y']
        
    def __len__(self):
        return len(self.x_vals)
    
    def __iter__(self):
        for x in self.x_vals:
            for y in self.y_vals:
                yield (x,y)

    # returns an x,y points for a trajecory of a given index.
    def get_trajectory(self,index):
        x = self.x_vals[index]
        y = self.y_vals[index]
        return x,y


    def plot_trajectory(self,index):
        x,y = self.get_trajectory(index)
        fig, ax1 = plt.subplots()
        ax1.plot(x, y)
        fig.tight_layout()
        plt.show()
    
    def generate_psf(self,x,y,kernel_size):
        psf = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(len(x)):
            psf[center - round(y[i]), center + round(x[i])] += 1
        return psf

    # Generates PSF according to given index of trajectory.
    def generate_kernel(self,index, kernel_size = 20 ,plotshow = False):
        x, y = self.get_trajectory(index)
        kernel = self.generate_psf(x, y,kernel_size)
        
        if plotshow == True:
            plt.imshow(kernel, cmap='gray')
            plt.show()
            
        return kernel
