import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import fftpack,signal
import scipy.signal
import skimage.measure
import sklearn
import sklearn.decomposition
import sklearn.neighbors

class KernelEsimator:
    
    def __init__(self,image,alpha):
        self.base_image = image
        self.alpha = alpha
        self.high_resolution_image = self.upsample_image(image)
        self.low_resolution_image =   self.downsample_image(image) 
        
    def downsample_image(self,image):
        (mat_shape_x, mat_shape_y) = image.shape
        new_size =(int(mat_shape_x / self.alpha), int(mat_shape_y / self.alpha))
        downsampled = np.zeros(new_size)
        for i in range(new_size[0]):
            for j in range(new_size[1]):
                downsampled[i, j] = image[self.alpha * i, self.alpha * j]
        return downsampled

    def upsample_image(self,image):
        (mat_shape_x, mat_shape_y) = image.shape
        new_size = (int(mat_shape_y * self.alpha), int(mat_shape_x * self.alpha))
        upsampled_filtered_image = cv2.resize(image, dsize=new_size, interpolation=cv2.INTER_CUBIC)
        return upsampled_filtered_image
    
        
    def esimator_kernel(self,initial_k,**hp):
        
        #create reduced patches
        patches = self.__create_patches__(hp['patch_size'])
        transformed_patches = sklearn.decomposition.PCA(n_components=hp['reduced_patch_size']).fit_transform(patches)        
        
        Rj = self.__generating_Rj__()

        C = self.__generate_generalization_term__()
        C_squared = C.T @ C
        
        estimated_kernels = []
        for i in range(hp["iterations"]):
            print(f"iterations {i+1}")
            guess_k = self.esimator_iteration_phase(r_patches,q_patches,Rj,guess_k,C_squared, hp["num_neighbors"],hp["sigma"])
            estimated_kernels.append(guess_k.reshape((hp['patch_size'], hp['patch_size'])))
        
        return estimated_kernels

    def esimator_iteration_phase(self,r_patches,q_patches,Rj,guess_k,C_squared,num_neighbors,sigma):
        raise NotImplementedError("")
            
        #down-sample example patches
        r_alpha_patches = []
        for j, patch in enumerate(r_patches):
            curr_patch_alpha = Rj[j] @ k
            r_alpha_patches.append(curr_patch_alpha)
        
        neighbors_weights = self.__find_nearest_neighbors__(r_alpha_patches,q_patches,num_neighbors,sigma)
        curr_k = self.__calculate_upadet_k__(guess_k,neighbors_weights,Rj,C_squared,q_patches,sigma)
        
        return curr_k
      
    
######################################### Side functions for simplification ##################################################################################################################
    def __generate_generalization_term__(self):
        raise NotImplementedError("")
    
    
    def __gaussian_distance__(self,x, y, sigma):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

    def __create_patches__(self,patch_size):
        radius = patch_size//2
        height, width = self.low_resolution_image.shape
        padded_image = np.pad( self.low_resolution_image, radius, mode='reflect')
        patches = np.zeros((height * width, patch_size ** 2, ))
        for i in range(radius, height + radius):
            for j in range(radius, width + radius):
                patch = padded_image[i - radius:i + radius + 1, j - radius:j + radius + 1]
                patches[(i - radius) * width + (j - radius), :] = patch.flatten()
        return patches
    
    
    def __generating_Rj__(self,patches):
        raise NotImplementedError("")
        Rj = []
        for r in patches:
            r_circulant = scipy.linalg.circulant(r)
            curr_Rj = downsample_shrink_matrix_1d(r_circulant, self.alpha ** 2)
            Rj.append(curr_Rj)
        
        return Rj
    

    def __find_nearest_neighbors__(self,r_alpha_patches,q_patches,num_neighbors,sigma):
        raise NotImplementedError("")
        
        r_alpha_patches = np.array(r_alpha_patches)
        neighbors_weights = np.zeros((len(q_patches), len(r_alpha_patches)))
        
        tree = sklearn.neighbors.BallTree(r_alpha_patches, leaf_size=2)      
        for i, q in enumerate(q_patches):
            representative_patch = np.expand_dims(q, 0)
            _, neighbor_indices = tree.query(representative_patch, k=num_neighbors)
            for index in neighbor_indices:
                neighbors_weights[i, index] = self.__gaussian_distance__(q, r_alpha_patches[index], sigma)
        return neighbors_weights

    def __calculate_upadet_k__(self,guess_k,neighbors_weights,Rj,C_squared,q_patches,sigma):
        raise NotImplementedError("")
        
        sum_left = np.zeros((guess_k.shape[0], guess_k.shape[0]))
        sum_right = np.zeros_like(guess_k)

        for i in range(neighbors_weights.shape[0]):
            for j in range(neighbors_weights.shape[1]):
                if not neighbors_weights[i, j]:
                    continue
                R_squared = Rj[j].T @ Rj[j]

                sum_left += neighbors_weights[i, j] * (R_squared) + (C_squared)
                sum_right += neighbors_weights[i, j] * (Rj[j].T @ q_patches[i])
        
        
        curr_k = np.linalg.inv((1 / (sigma ** 2)) * sum_left) @ sum_right
        
        return curr_k

    



    



