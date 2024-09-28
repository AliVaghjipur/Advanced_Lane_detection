#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Applying Distortion Correction:
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


# In[2]:


# Load the camera calibration data
calibration_data = np.load('calibration_data.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']


# In[3]:


# Function to undistort an image
def undistort_image(image, camera_matrix, dist_coeffs):
    undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)
    return undistorted_img


# In[6]:


# Path to your test images (replace with actual image paths)
test_images = glob.glob('TestImages/*.jpg') #Check the path correctly

# Process each image
for fname in test_images:
    print(f"Processing image: {fname}")
    img = cv2.imread(fname)
    
    if img is None:
        print(f"Failed to load image: {fname}")
        continue

    # Apply distortion correction
    undistorted_img = undistort_image(img, camera_matrix, dist_coeffs)

    # Display the original and undistorted images side by side
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax2.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted Image')
    plt.show()


# In[ ]:




