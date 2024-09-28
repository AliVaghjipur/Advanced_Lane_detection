#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from LaneFittingWithWindow import find_lane_pixels
from LaneFittingWithWindow import fit_polynomial
# In[4]:


# Load our image - this should be a new frame since last time!
binary_warped = mpimg.imread('warped-example.jpg')

# Polynomial fit values from the previous frame
# Make sure to grab the actual values from the previous step in your project!
prev_left_fit = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
prev_right_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])


# In[5]:


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # Fit a second-order polynomial to the lane lines using the pixel positions
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate the x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty, left_fit, right_fit


# In[6]:


def search_around_poly(binary_warped, prev_left_fit, prev_right_fit, margin=100):
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Set the area of search based on activated x-values within the margin
    left_lane_inds = ((nonzerox > (prev_left_fit[0] * (nonzeroy ** 2) + prev_left_fit[1] * nonzeroy + prev_left_fit[2] - margin)) & 
                      (nonzerox < (prev_left_fit[0] * (nonzeroy ** 2) + prev_left_fit[1] * nonzeroy + prev_left_fit[2] + margin)))
    
    right_lane_inds = ((nonzerox > (prev_right_fit[0] * (nonzeroy ** 2) + prev_right_fit[1] * nonzeroy + prev_right_fit[2] - margin)) & 
                       (nonzerox < (prev_right_fit[0] * (nonzeroy ** 2) + prev_right_fit[1] * nonzeroy + prev_right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fallback to sliding window search if not enough points are found
    if len(leftx) < 500 or len(rightx) < 500:
        print("Not enough points found in previous lane. Falling back to sliding window search.")
        return fit_polynomial(binary_warped)  # Call sliding window as fallback
    
    # Fit new polynomials based on the pixel positions found
    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])

    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])

    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return left_fit, right_fit, result

# Run image through the pipeline
left_fit, right_fit, result = search_around_poly(binary_warped, prev_left_fit, prev_right_fit)

# View your output
plt.imshow(result)
plt.show()


#Measuring curvature of the lane line using formula in AutonomousDrivingCourse/LaneLines/RadiusOfCurvatureFormula.png

