import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from tensorflow.keras.models import load_model
import os

class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    # Ensure image is in the correct format
    if image is None:
        return None
    
    # Get original image dimensions
    height, width = image.shape[:2]
    
    # Resize input image for model
    small_img = cv2.resize(image, (160, 80))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]
    
    # Get prediction
    prediction = model.predict(small_img)[0] * 255
    lanes.recent_fit.append(prediction)
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]
    
    # Calculate average fit
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)
    
    # Create blank image with same dimensions as input
    blanks = np.zeros_like(image)
    
    # Resize prediction to match input image dimensions
    lane_drawn = cv2.resize(lanes.avg_fit, (width, height))
    
    # Create colored lane overlay
    lane_image = np.zeros_like(image)
    lane_image[:, :, 1] = lane_drawn  # Green channel
    
    # Combine images
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)
    return result


if __name__ == '__main__':
    model = load_model('../models/full_CNN_model.h5')
    lanes = Lanes()
    vid_output = '../data/output_video.mp4'
    clip1 = VideoFileClip("../data/input_video.mp4")
    vid_clip = clip1.fl_image(road_lines)
    vid_clip.write_videofile(vid_output, audio=False)
