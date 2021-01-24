### Wet HW2 DIP ###
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import fftpack
###########################################################################
###########################################################################
###########################################################################
image_file = 'DIPSourceHW2.png'
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
def plot_image(image_, name_: str):
    plt.figure()
    plt.imshow(image_, cmap="gray")
    plt.title(name_)
    plt.show(block=0)
    plt.savefig(name_)
    return
# ################## 1 ##################
height = 346
width = 550
epsilon = 1e-9
def PSF_Gauss(std):
    x = np.linspace(-10, 10, width)
    y = np.linspace(-10, 10, height)
    X, Y = np.meshgrid(x, y)
    exp_ = np.exp(-1 * (X ** 2 + Y ** 2) / (2 * (std * std)))
    return exp_ / (2 * np.pi * std)
    
PFS_l_gauss = PSF_Gauss(0.25)
PFS_h_gauss = PSF_Gauss(0.04)
plot_image(PFS_l_gauss, '1 - PFS_l_gauss')
plot_image(PFS_h_gauss, '1 - PFS_h_gauss')
# ################## 2 ##################
def PSF_box(box_size):
    h_ = int(height / 2)
    w_ = int(width / 2)
    d_ = int(box_size / 2)
    PSF_box_ = np.zeros((height, width))
    PSF_box_[h_ - d_:h_ + d_, w_ - d_:w_ + d_] = 1 / (box_size ** 2)
    return PSF_box_

PSF_l_box = PSF_box(6)
PSF_h_box = PSF_box(3)
plot_image(PSF_l_box, '2 - PSF_l_box')
plot_image(PSF_h_box, '2 - PSF_h_box')

# ################## 3 ##################
def resolution_image(image_, psf_):
    f_image = fftpack.fftshift(fftpack.fftn(image_))
    k_ = fftpack.fftshift(fftpack.fftn(psf_))
    return np.abs(fftpack.fftshift(fftpack.ifftn(f_image * k_)))

gauss_l = resolution_image(image, PFS_l_gauss)
gauss_h = resolution_image(image, PFS_h_gauss)
box_l = resolution_image(image, PSF_l_box)
box_h = resolution_image(image, PSF_h_box)
plot_image(gauss_l, '3 - gauss_l')
plot_image(gauss_h, '3 - gauss_h')
plot_image(box_l, '3 - box_l')
plot_image(box_h, '3 - box_h')

# ################## 4 ##################
def blur_kernel_k(psf_l_, psf_h_):
    PSF_l_F = fftpack.fftshift(fftpack.fftn(psf_l_))
    PSF_h_F = fftpack.fftshift(fftpack.fftn(psf_h_))
    PSF_h_F[np.abs(PSF_h_F) < epsilon] = epsilon
    return np.abs(fftpack.fftshift(fftpack.ifftn(PSF_l_F / PSF_h_F)))

k_gauss = blur_kernel_k(PFS_l_gauss, PFS_h_gauss)
k_box = blur_kernel_k(PSF_l_box, PSF_h_box)
plot_image(k_gauss, '4 - k_gauss')
plot_image(k_box, '4 - k_box')
# ################## 5 - wiener ##################

def wiener(high_res_image, k_):
    L_ = fftpack.fftshift(fftpack.fftn(high_res_image))
    K_ = fftpack.fftshift(fftpack.fftn(k_))
    K_ = K_ + epsilon
    H_ = L_ / K_
    return np.abs(fftpack.fftshift(fftpack.ifftn(H_)))
wiener_gauss = wiener(gauss_l, k_gauss)
wiener_box = wiener(box_l, k_box)
plot_image(wiener_gauss, '5 - wiener_gauss')
plot_image(wiener_box, '5 - wiener_box')

# ################## 5 - TV ##################
def TV(high_res_image, k_):
    d_op = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    D_ = fftpack.fftshift(fftpack.fftn(d_op, shape=high_res_image.shape))
    K_ = fftpack.fftshift(fftpack.fftn(k_))
    L_ = fftpack.fftshift(fftpack.fftn(high_res_image))
    filter_ = K_ / (K_ ** 2 + 0.3 * D_ ** 2)
    filter_ = np.abs(filter_)
    filter_ = filter_ + epsilon
    H_ = L_ * filter_
    return np.abs(fftpack.ifftn(H_))

TV_gauss = TV(gauss_l, k_gauss)
TV_box = TV(box_l, k_box)
plot_image(TV_gauss, '5 - TV_gauss')
plot_image(TV_box, '5 - TV_box')

# ################## 6 ##################
bilinear_gauss = cv2.resize(gauss_l, dsize=None, fx=2, fy=2)
bilinear_box = cv2.resize(box_l, dsize=None, fx=2, fy=2)
plot_image(bilinear_gauss, '6 - bilinear_gauss')
plot_image(bilinear_box, '6 - bilinear_box')
bicubic_gauss = cv2.resize(gauss_l, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
bicubic_box = cv2.resize(box_l, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
plot_image(bicubic_gauss, '6 - bicubic_gauss')
plot_image(bicubic_box, '6 - bicubic_box')

# ################################################
plt.show()
# ################################################
# ################################################
# ################################################