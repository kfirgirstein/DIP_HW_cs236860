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
        
    # Generates PSF according to given index of trajectory.
    def generate_kernel(self,index, kernel_size = 13,zfactor = 1,show = False):
        x, y = self.get_trajectory(index)
        center_shift = (kernel_size-1)/2
        kernel = np.zeros((kernel_size,kernel_size),int)

        for k in range(len(x)):
            kernel_row = int(round(center_shift+x[k]*zfactor))
            kernel_col = int(round(center_shift-y[k]*zfactor))
            if kernel_col<kernel_size and kernel_row<kernel_size:
                kernel[kernel_col, kernel_row] = kernel[kernel_col, kernel_row] + 1

        if show == True:
            plt.imshow(kernel, cmap='gray')
            plt.show()
            
        return kernel
