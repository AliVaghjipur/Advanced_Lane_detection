#!/usr/bin/env python
# coding: utf-8

# In[48]:


# Step 3: Create a Thresholded Binary Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from DistortionCorrection import undistort_image


# In[49]:


# Load the camera calibration data
calibration_data = np.load('calibration_data.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']


# In[50]:


# Function to apply Sobel operator to detect edges
#Sobel thresholding:  Sobel operator is used to detect edges (primarily in the x-direction) by calculating the gradient of pixel intensity.

def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel in x or y direction
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Apply threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output

# Function to apply a threshold in a specific color channel (like S channel in HLS)
#Color thresholding: Color threshold is applied using the S channel from the HLS color space to capture lane lines more effectively.

def color_threshold(img, thresh_min=170, thresh_max=255):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]  # S channel from HLS

    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1
    return binary_output

# Function to combine gradient and color thresholds
#Combining thresholds: The gradient and color thresholds are combined to produce a binary image.
def combined_threshold(img):
    # Apply Sobel in x direction
    gradx = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)
    
    # Apply color threshold (S channel)
    s_binary = color_threshold(img, thresh_min=170, thresh_max=255)
    
    # Combine the two binary images
    combined_binary = np.zeros_like(gradx)
    combined_binary[(gradx == 1) | (s_binary == 1)] = 1
    return combined_binary


# In[51]:


# Apply the thresholds to a test image
test_image = cv2.imread('TestImages/test3.jpg')
undistorted_test_image = undistort_image(test_image, camera_matrix, dist_coeffs)

# Get the binary image
binary_image = combined_threshold(undistorted_test_image)

# Display the binary image
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Thresholded Image')
plt.show()


# In[54]:


# Step 4: Apply Perspective Transform
# Define source and destination points
def define_perspective_transform(src_img):
    # Define the source points in the original image
    src = np.float32([[200, src_img.shape[0]], [1100, src_img.shape[0]], [595, 450], [685, 450]])
    
    # Define the destination points for the top-down view
    dst = np.float32([[300, src_img.shape[0]], [980, src_img.shape[0]], [300, 0], [980, 0]])

    return src, dst

def warp_perspective(img, src, dst):
    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Warp the image using the perspective transform matrix
    warped_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    
    return warped_img, Minv


# In[53]:


# Example usage
# Load a binary image
#binary_image = cv2.imread('path_to_binary_image.jpg', cv2.IMREAD_GRAYSCALE)  # Ensure binary image is loaded correctly

# Define the source and destination points
src, dst = define_perspective_transform(binary_image)

# Apply the perspective transform
warped_image, Minv = warp_perspective(binary_image, src, dst)

# Display the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(binary_image, cmap='gray')
ax1.set_title('Original Binary Image')
ax2.imshow(warped_image, cmap='gray')
ax2.set_title('Perspective Warped Image')
plt.show()


# In[47]:


# For Evaluating Perspective Transform
#src = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])
#dst = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])

#def evaluate_perspective_transform(original_image, warped_image):
    # Display the original and warped images side by side for visual inspection
#    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
#    ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#    ax1.set_title('Original Image')
#    ax2.imshow(warped_image, cmap='gray')
#    ax2.set_title('Warped Image')
#    plt.show()

# Load an example image and apply the transformation
#original_image = cv2.imread('TestImages/test3.jpg')
#binary_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for binary processing
#warped_image = warp_perspective(binary_image, src, dst)

#evaluate_perspective_transform(original_image, warped_image)


# In[ ]:

def preprocess_image(img, camera_matrix, dist_coeffs, src, dst):
    """
    This function combines undistortion, thresholding, and perspective transform steps
    to create a binary warped image for lane detection.
    
    Parameters:
        img: Input image (color image)
        camera_matrix: Camera matrix from calibration
        dist_coeffs: Distortion coefficients from calibration
        src: Source points for perspective transform
        dst: Destination points for perspective transform
    
    Returns:
        warped_binary: Perspective-transformed binary image
    """
    
    # 1. Undistort the image
    undistorted_img = undistort_image(img, camera_matrix, dist_coeffs)

    # 2. Apply combined gradient and color thresholds
    binary_image = combined_threshold(undistorted_img)

    # 3. Apply perspective transform
    warped_binary, Minv = warp_perspective(binary_image, src, dst)

    return warped_binary, Minv

# Example usage
# Load the camera calibration data
calibration_data = np.load('calibration_data.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Load the test image
test_image = cv2.imread('TestImages/test2.jpg')

# Define source and destination points for perspective transform
src, dst = define_perspective_transform(test_image)

# Process the image
binary_warped, Minv = preprocess_image(test_image, camera_matrix, dist_coeffs, src, dst)

# Display the final binary warped image
plt.imshow(binary_warped, cmap='gray')
plt.title('Binary Warped Image')
plt.show()