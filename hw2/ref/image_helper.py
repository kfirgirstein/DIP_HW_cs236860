import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import fftpack
import scipy.signal

def plot_image(image_, name_: str,path_prefix=None):
    plt.figure()
    plt.imshow(image_, cmap="gray")
    plt.title(name_)
    plt.show(block=0)
    if path_prefix:
        plt.savefig(path_prefix + name_)
    return


def gaussian_kernel(size, std=1):
    #cv2.GaussianBlur(noisy_image, (BLUR_FILTER_SIZE, BLUR_FILTER_SIZE), BLUR_FILTER_STD)
    edge = size // 2
    ax = np.linspace(-edge, edge, num=size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * std **2))
    return kernel / kernel.sum()

def wiener_filter(image, psf, k=0.01):
    image_dft = fftpack.fft2(image)
    psf_dft = fftpack.fft2(fftpack.ifftshift(psf))
    filter_dft = np.conj(psf_dft) / (np.abs(psf_dft) ** 2 + k)
    recovered_dft = image_dft * filter_dft
    return np.real(fftpack.ifft2(recovered_dft))

def estimate_filter(psf_l_, psf_h_,epsilon=1e-9):
    raise NotImplementedError("")

    
####### ref #########
def PSF_Gauss(width,height,std):
    x = np.linspace(-10, 10, width)
    y = np.linspace(-10, 10, height)
    X, Y = np.meshgrid(x, y)
    exp_ = np.exp(-1 * (X ** 2 + Y ** 2) / (2 * (std * std)))
    return exp_ / (2 * np.pi * std)

def PSF_box(width,height,box_size):
    h_ = int(height / 2)
    w_ = int(width / 2)
    d_ = int(box_size / 2)
    PSF_box_ = np.zeros((height, width))
    PSF_box_[h_ - d_:h_ + d_, w_ - d_:w_ + d_] = 1 / (box_size ** 2)
    return PSF_box_


def blur_kernel_k(psf_l_, psf_h_,epsilon=1e-9):
    PSF_l_F = fftpack.fftshift(fftpack.fftn(psf_l_))
    PSF_h_F = fftpack.fftshift(fftpack.fftn(psf_h_))
    PSF_h_F[np.abs(PSF_h_F) < epsilon] = epsilon
    return np.abs(fftpack.fftshift(fftpack.ifftn(PSF_l_F / PSF_h_F)))

def wiener(high_res_image, k_,epsilon=1e-9):
    L_ = fftpack.fftshift(fftpack.fftn(high_res_image))
    K_ = fftpack.fftshift(fftpack.fftn(k_))
    K_ = K_ + epsilon
    H_ = L_ / K_
    return np.abs(fftpack.fftshift(fftpack.ifftn(H_)))

    
def TV(high_res_image, k_,epsilon=1e-9):
    d_op = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    D_ = fftpack.fftshift(fftpack.fftn(d_op, shape=high_res_image.shape))
    K_ = fftpack.fftshift(fftpack.fftn(k_))
    L_ = fftpack.fftshift(fftpack.fftn(high_res_image))
    filter_ = K_ / (K_ ** 2 + 0.3 * D_ ** 2)
    filter_ = np.abs(filter_)
    filter_ = filter_ + epsilon
    H_ = L_ * filter_
    return np.abs(fftpack.ifftn(H_))