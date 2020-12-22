import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import signal
from scipy import fftpack
import scipy.io




    
def main():
    my_Im = Image_cl()

    my_applier = Blurr_Applier(100)
    my_applier.generate_PSFs()
    my_applier.apply_blurr()
    print("Plot Trajectories, PSFs and Blurred images")
    #my_applier.plot_batches_of_5()

    # Get a list of blurred images
    blurred_images = my_applier.get_blurred_images()

    # Fix the blurred images
    my_fixer = Blurr_Fixer(blurred_images,p=10,ifft_scale=995,
                               original_size=256, margin=6)
    fixed = my_fixer.fix_blurr()
    #my_fixer.show_Fixed()

    print("PLot n to PSNR graph")

    num_samples = list()
    PSNR_results = list()
    fixed_images = list()

    # Iterate over different number of blurred images,
    # Calc the PSNR and show the result
    for n in range(1,101,1):
        print("Deblurring for ",n," samples...")
        num_samples.append(n)
        my_applier = Blurr_Applier(n)
        my_applier.generate_PSFs()
        my_applier.apply_blurr()

        blurred_images = my_applier.get_blurred_images()

        my_fixer = Blurr_Fixer(blurred_images, p=10,ifft_scale=995,
                               original_size=256, margin=6)
        fixed = my_fixer.fix_blurr()
        fixed_images.append(fixed)
        my_calc = PSNR_calculator(my_Im.get_image(), fixed)
        PSNR_results.append(my_calc.evaluate_PSNR())

    plt.plot(num_samples, PSNR_results)
    plt.xlabel("Number of blurred samples")
    plt.ylabel("PSNR [dB]")
    plt.savefig("C:\\Users\\Pavel\\Desktop\\DIP_hw1_repo\\results\\psnr_graph\\psnr_graph.png")
    plt.show()

    # save to file PSNR + images
    for i in range(len(fixed_images)):
        cropped = fixed_images[i]
        plt.imshow(cropped, cmap='gray')
        k = i+1
        plt.title("n={0}, PSNR={1}".format(k,PSNR_results[i]), fontsize=10)
        plt.tick_params(labelbottom=False)
        plt.tick_params(labelleft=False)
        path = "C:\\Users\\Pavel\\Desktop\\DIP_hw1_repo\\results\\psnr_images\\"
        full_path = path + str(i) + '.png'
        plt.savefig(full_path)


    for i in range(len(fixed_images)):
        plt.subplot(10, 10, i+1)
        cropped = fixed_images[i][25:100, 100:175]
        plt.imshow(cropped, cmap='gray')
        k = i+1
        plt.ylabel("n=%i" % k, fontsize=5)
        plt.tick_params(labelbottom=False)
        plt.tick_params(labelleft=False)
    plt.show()

    # Show first and last iteration for comparison.
    #plt.subplot(1, 3, 1)
    #cropped = blurred_images[0]
    #plt.imshow(cropped, cmap='gray')
    #plt.title("First Blurred image")
    #plt.subplot(1, 3, 2)
    #cropped = fixed_images[0]
    #plt.imshow(cropped, cmap='gray')
    #plt.title("First iteration")
    #plt.subplot(1, 3, 3)
    #cropped = fixed
    #plt.imshow(cropped, cmap='gray')
    #plt.title("100th iteration")
    #plt.show()


if __name__ == '__main__':
    main()