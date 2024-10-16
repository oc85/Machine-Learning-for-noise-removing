import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from tqdm import tqdm
import shutil
from patchify import patchify

# Constants
SOURCE_DIR = r'...\ground_truth'
SOURCE_DATA = os.path.join(SOURCE_DIR, 'images')
SOURCE_MASKS = os.path.join(SOURCE_DIR, 'masks')
IMAGE_RESIZE = 1024
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
SE2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
SE3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Directories for training data
TRAIN_DATA_DIR = os.path.join(SOURCE_DIR, 'training')
TRAIN_IMAGES_DIR = os.path.join(TRAIN_DATA_DIR, 'train_data')
TRAIN_LABELS_DIR = os.path.join(TRAIN_DATA_DIR, 'train_labels')

# Function to create directories
def create_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, mode=0o777)

# Prepare directories
create_directory(TRAIN_DATA_DIR)
create_directory(TRAIN_IMAGES_DIR)
create_directory(TRAIN_LABELS_DIR)

# Process input images 
def process_images(source_data, target_dir, image_resize):
    sorted_folders = sorted(os.listdir(source_data))
    for idx, filename in enumerate(tqdm(sorted_folders, desc="Processing images")):
        filepath = os.path.join(source_data, filename)
        if filepath.endswith('.jpg'):
            # Read and resize image
            image = mpimg.imread(filepath)
            resized_image = cv2.resize(image, (image_resize, image_resize))

            # Patchify the image
            patches = patchify(resized_image, (256, 256), step=256)
            patches_reshaped = patches.reshape((-1, 256, 256, 3))

            # Save patches
            for patch_idx, patch in enumerate(patches_reshaped):
                patch_filename = os.path.join(target_dir, f'data_{idx}_{patch_idx}.jpg')
                cv2.imwrite(patch_filename, patch)

# Process target images 
i,j=0,0
sorted_list_folder= sorted(os.listdir(source_masks))
for idx2 in tqdm(sorted_list_folder):
          filepath = os.path.join(source_masks,idx2)
          #print(filepath)
          if filepath.endswith(('.tif')):
              file = mpimg.imread(filepath)
              file=cv2.resize(file,(image_resize,image_resize)) # 256x12
              file=(file[:,:]-file[:,:].min())/(file[:,:].max()-file[:,:].min())
              file=np.uint8(file>0)
              patches = patchify(file, (256, 256), step=256)
              patches = np.array(patches)
              patches = np.stack((patches,)*3,axis=-1)
              patches_reshaped = np.reshape(patches, (patches.shape[0]* patches.shape[1], 256,256,3) )

              #for ii in range(patches_reshaped.shape[0]):
              for ii in range(patches_reshaped.shape[0]):
                  #filename=source_data +r'\test_LR\data_LR_'+str(i)+'_'+str(ii)+'.tif'
                  filename=dirpathHR+r'\mask_'+str(i)+'_'+str(ii)+'.jpg'
                  cv2.imwrite(filename, 256*patches_reshaped[ii])
              i=i+1
          patchL=patches_reshaped[0]
# Execute image processing
process_images(SOURCE_DATA, TRAIN_IMAGES_DIR, IMAGE_RESIZE)
#################################################################################

# Function to process and save patches from images
def process_masks(source_masks, target_dir, image_resize):
    sorted_folders = sorted(os.listdir(source_masks))
    for idx, filename in enumerate(tqdm(sorted_folders, desc="Processing masks")):
        filepath = os.path.join(source_masks, filename)
        if filepath.endswith('.tif'):
            # Read and resize image
            image = mpimg.imread(filepath)
            resized_image = cv2.resize(image, (image_resize, image_resize))

            # Normalize and binarize
            normalized_image = (resized_image - resized_image.min()) / (resized_image.max() - resized_image.min())
            binary_image = np.uint8(normalized_image > 0)

            # Patchify the image
            patches = patchify(binary_image, (256, 256), step=256)
            patches_reshaped = patches.reshape((-1, 256, 256, 1))
            patches_reshaped = np.repeat(patches_reshaped, 3, axis=-1)  # Repeat channels to make RGB

            # Save patches
            for patch_idx, patch in enumerate(patches_reshaped):
                patch_filename = os.path.join(target_dir, f'mask_{idx}_{patch_idx}.jpg')
                cv2.imwrite(patch_filename, 256 * patch)

# Execute mask processing
process_masks(SOURCE_MASKS, TRAIN_LABELS_DIR, IMAGE_RESIZE)
