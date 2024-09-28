# Advanced_Lane_detection

# Lane Detection Project

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Overview](#algorithm-overview)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Introduction
The Lane Detection Project is aimed at developing an effective lane detection algorithm capable of accurately detecting lane markings in various driving conditions, including sharp curves and high-contrast lighting. The project utilizes computer vision techniques to process video frames and overlay detected lanes on the original video.

## Getting Started
This project processes input video files, applies image preprocessing techniques, and outputs a video with detected lanes highlighted.

### Prerequisites
- Python 3.x
- OpenCV
- NumPy
- Matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/LaneDetectionProject.git
   cd LaneDetectionProject

## Usage
Place your input video file in the place of the 'project_video.mp4' in the file FinalLaneDetection.ipynb
Run the lane detection algorithm:
View the output video in the same directory

## Algorithm Overview
The lane detection algorithm includes the following key steps:

Undistortion: Correcting lens distortion using camera calibration data.
Thresholding: Applying Sobel and color thresholding to isolate lane markings.
Region of Interest (ROI): Masking to focus on the road area.
Perspective Transform: Warping the image to a bird’s-eye view.
Hough Transform: Detecting lines in the binary image.
Lane Curvature and Offset Calculation: Estimating the curvature of the detected lanes and the vehicle's position within the lane.
Output Video: Overlaying detected lanes on the original video.

## Results
Here are some results from the lane detection algorithm:
![Advanced_Lane_detection](Output
