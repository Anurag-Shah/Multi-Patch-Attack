# Multi-Patch-Attack
Adversarial patch attack using multiple patches to successfully attack the Minority Reports Defense and avoid Wavelet Patch Detection. 

# Setup:
Requires pytorch 1.10.

For the ImageNet dataset, download https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar (you may need to create an account). This is the validation dataset of 50,000 images, with 50 images in each of 1,000 classes. Then, run the script valprep.sh while you are in the same directory as the images. This will divide the validation images into the 1,000 classes. Specify the directory within the python script.

# Contributors
Anurag Shah

# Supervisors
Dr. Gustavo Rodriguez-Rivera, Maxwell Jacobson
