import cv2
import numpy as np
import scipy
from scipy import fftpack, signal
from sklearn.neighbors import ball_tree
from scipy.sparse import spdiags
import matplotlib.pyplot as plt


def sinc_psf(window_size=15, edge=15/4):
    x = np.expand_dims(np.linspace(-edge, edge, window_size),axis=0)
    s = np.sinc(x.T @ x)
    s = s / s.sum()
    return s
      
# based on tutorial 2
def gaussian_psf(size, std=1):
    edge = size // 2
    ax = np.linspace(-edge, edge, num=size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * std **2))
    return kernel / kernel.sum()


# extracting patches based on tutorial 8
def make_patches(image, patch_size, step_size=1):
    radius = patch_size // 2
    height, width = image.shape

    padded_image = np.pad(image, radius, mode='reflect')
    patches = []
    for i in range(radius, height + radius, step_size):
        for j in range(radius, width + radius, step_size):
            patch = padded_image[i - radius:i + radius + 1, j - radius:j + radius + 1]
            patches.append(patch.flatten())
    
    return np.array(patches)


def create_laplacian(patch_size):

    # create laplacian kernel
    data = np.array([[4]*(patch_size**2), [-1]*(patch_size**2), [-1]*(patch_size**2), [-1]*(patch_size**2), [-1]*(patch_size**2)])

    # create the diagonalized matrix of the laplacian kernel for matrix multiplication
    diags = np.array([0, -1, 1, patch_size, -patch_size])
    spd = spdiags(data, diags, patch_size**2, patch_size**2).toarray()
    for i in range(1, patch_size):
        spd[patch_size * i - 1][patch_size * i] = 0
        spd[patch_size * i][patch_size * i - 1] = 0
    
    return spd
    

def compute_weights_numerator(q_patches, r_alphas, sigma):
    return np.exp(-0.5 * (np.linalg.norm(q_patches-r_alphas, axis=1)**2) / (sigma**2))


def compute_k(downsampled_image, alpha, iter_num, patch_size=5, num_neighbours=5, sigma=0.06):
    
    # create r and q patches 
    step_size_factor = 1
    r_patch_size = patch_size * alpha
    q_patches = make_patches(downsampled_image, patch_size, step_size=step_size_factor)
    r_patches = make_patches(downsampled_image, r_patch_size, step_size=step_size_factor * alpha)
    
    # initialize k with a delta function
    k = np.expand_dims(signal.unit_impulse((r_patch_size,r_patch_size), idx='mid').flatten(), axis=1) #changed backwards

    # create laplacian matrix
    C = create_laplacian(r_patch_size) #changed. was patch_size instead
    CTC = C.T @ C

    # create Rj matrices for each patch rj
    Rs = create_R_mats(r_patches, r_patch_size, alpha**2)
    Rs_np = np.array(Rs)
    Rs_np_T = np.swapaxes(Rs_np, 1, 2) # transpose each Rj
    Rs_T_Rs_np = Rs_np_T @ Rs_np
        
    for _ in range(iter_num):
        # compute patches in the coarse image

        r_alphas = (Rs @ k).squeeze()
        # find nearest neighbours for each qi (NLM trick)
        tree = ball_tree.BallTree(r_alphas, leaf_size=2)
        _, neighbors_idxs = tree.query(q_patches, k=num_neighbours)
        # compute weights of nearest neighbours
        weights = np.zeros((q_patches.shape[0], r_alphas.shape[0])) 
        for i in range(q_patches.shape[0]):
            weights[i, neighbors_idxs[i]] = compute_weights_numerator(q_patches[i], r_alphas[neighbors_idxs[i]], sigma)
        weights_sum = weights.sum(axis=1)

        # normalize weights
        weights = np.nan_to_num(np.divide(weights , np.expand_dims(weights_sum, -1)))

        # compute the components of the next k
        sum_wRTR_CTC = np.zeros((k.shape[0],k.shape[0]))
        sum_wRTq = np.zeros(k.shape)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                if weights[i, j]:
                    sum_wRTR_CTC += weights[i,j] * Rs_T_Rs_np[j]
                    sum_wRTq += weights[i,j] * Rs_np_T[j] @ np.expand_dims(q_patches[i],-1)

        # compute the next k
        inv_wRTR_CTC = np.linalg.inv(sum_wRTR_CTC + CTC) # CTC is outside the sum
        k = inv_wRTR_CTC @ sum_wRTq
        print(f'mean: {k.mean()} std: {k.std()}')

    # return as a 2D kernel
    return k.reshape((r_patch_size, r_patch_size))


# based on tutorial 7
def wiener_filter(image, psf, k=0.1):
    image_dft = fftpack.fft2(image)
    psf_dft = fftpack.fft2(fftpack.ifftshift(psf), shape=image_dft.shape)
    filter_dft = np.conj(psf_dft) / (np.abs(psf_dft) ** 2 + k)
    recovered_dft = image_dft * filter_dft
    return np.real(fftpack.ifft2(recovered_dft))


def create_R_mats(r_patches, r_patch_size, downsampling_scale):

    # create the R matrices for the r patches for matrix multiplication
    Rs = []
    for patch_i in range(r_patches.shape[0]):
        # refer to each one of the kernel's row as a small toeplitz
        toeplitz_list = []

        # add zeroed matrices for the indices that should be zeroed out below the diagonal
        ghost_toeplitz = np.zeros((r_patch_size,r_patch_size))
        for _ in range(r_patch_size//2):
            toeplitz_list.append(ghost_toeplitz)

        # create the small toeplitz matrices for each of the rows
        r_kernel = np.copy(r_patches[patch_i]).reshape([r_patch_size, r_patch_size])
        for i in range(r_patch_size):
            tiled = np.tile(r_kernel[i,:], (r_patch_size,1)).T
            diags = np.arange(-(r_patch_size//2),r_patch_size//2+1,1)
            small_toeplitz = spdiags(tiled, diags, r_patch_size, r_patch_size).toarray()
            toeplitz_list.append(small_toeplitz)

        # add zeroed matrices for the indices that should be zeroed out above the diagonal
        for _ in range(r_patch_size//2):
            toeplitz_list.append(ghost_toeplitz)

        # construct the big diagonalized matrix R that will correspond to the kernel r
        R_mat = np.zeros((r_patch_size**2, r_patch_size**2))
        for i in range(r_patch_size):
            for j in range(r_patch_size):
                toeplitz_small = toeplitz_list[r_patch_size-(i-j)-1]
                R_mat[i*r_patch_size:(i+1)*r_patch_size, j*r_patch_size: (j+1)*r_patch_size] = toeplitz_small
        
        # add and downscale it
        Rs.append(R_mat[::downsampling_scale,:])
        
    return Rs


if __name__ == "__main__":

    # read the given image from file
    image_tank = cv2.imread('DIPSourceHW2.png', cv2.IMREAD_GRAYSCALE)
    image_tank = image_tank / 255.0

    # create PSFs
    s_psf = sinc_psf()
    # 0.1 or 1 std might be better
    g_psf = gaussian_psf(size=16, std=1.)

    # init size, alpha and num of iterations
    alpha = 3
    iter_num = 15
    original_img_size = (image_tank.shape[1], image_tank.shape[0])
    new_img_size = (int(image_tank.shape[1]/alpha),int(image_tank.shape[0]/alpha))

    # use gaussian PSF to create downsampled and upsampled images
    blurred_image_gaussian = signal.convolve2d(image_tank, g_psf, mode='same', boundary='wrap')
    downsampled_image_gaussian = cv2.resize(blurred_image_gaussian, new_img_size, interpolation=cv2.INTER_NEAREST)
    upsampled_image_gaussian = cv2.resize(downsampled_image_gaussian, original_img_size, interpolation=cv2.INTER_CUBIC)

    # use sinc PSF to create downsampled and upsampled images
    blurred_image_sinc = signal.convolve2d(image_tank, s_psf, mode='same', boundary='wrap')
    downsampled_image_sinc = cv2.resize(blurred_image_sinc, new_img_size, interpolation=cv2.INTER_NEAREST)
    upsampled_image_sinc = cv2.resize(downsampled_image_sinc, original_img_size, interpolation=cv2.INTER_CUBIC)

    # compute k by each PSF
    k_gaussian = compute_k(downsampled_image_gaussian, alpha, iter_num)
    k_sinc = compute_k(downsampled_image_sinc, alpha, iter_num)
    
    # restore with wiener filter for each k for each PSF
    restored_gaussian_with_gaussian = wiener_filter(upsampled_image_gaussian, k_gaussian)
    restored_gaussian_with_sinc = wiener_filter(upsampled_image_gaussian, k_sinc)
    restored_sinc_with_sinc = wiener_filter(upsampled_image_sinc, k_sinc)
    restored_sinc_with_gaussian = wiener_filter(upsampled_image_sinc, k_gaussian)

    # restore with wiener filter for the true kernel
    restored_gaussian_with_true_kernel = wiener_filter(upsampled_image_sinc, g_psf)
    restored_sinc_with_true_kernel = wiener_filter(upsampled_image_gaussian, s_psf)

    # store low res images
    plt.imshow(downsampled_image_gaussian, cmap='gray')
    plt.savefig('downsampled_image_gaussian_plt.png')
    plt.cla()
    plt.imshow(downsampled_image_sinc, cmap='gray')
    plt.savefig('downsampled_image_sinc_plt.png')
    plt.cla()

    # store results and compute PSNR
    plt.imshow(restored_gaussian_with_gaussian, cmap='gray')
    plt.savefig('restored_gaussian_with_gaussian_plt.png')
    plt.cla()
    plt.imshow(restored_gaussian_with_sinc, cmap='gray')
    plt.savefig('restored_gaussian_with_sinc_plt.png')
    plt.cla()
    plt.imshow(restored_sinc_with_sinc, cmap='gray')
    plt.savefig('restored_sinc_with_sinc_plt.png')
    plt.cla()
    plt.imshow(restored_sinc_with_gaussian, cmap='gray')
    plt.savefig('restored_sinc_with_gaussian_plt.png')
    plt.cla()
    plt.imshow(restored_gaussian_with_true_kernel, cmap='gray')
    plt.savefig('restored_gaussian_with_true_kernel_plt.png')
    plt.cla()
    plt.imshow(restored_sinc_with_true_kernel, cmap='gray')
    plt.savefig('restored_sinc_with_true_kernel_plt.png')

    print("PSNR for gaussian on gaussian: {}".format(cv2.PSNR(np.float64(restored_gaussian_with_gaussian), np.float64(image_tank))))
    print("PSNR for sinc on gaussian: {}".format(cv2.PSNR(np.float64(restored_gaussian_with_sinc), np.float64(image_tank))))
    print("PSNR for sinc on sinc: {}".format(cv2.PSNR(np.float64(restored_sinc_with_sinc), np.float64(image_tank))))
    print("PSNR for gaussian on sinc: {}".format(cv2.PSNR(np.float64(restored_sinc_with_gaussian), np.float64(image_tank))))