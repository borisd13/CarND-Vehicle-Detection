# Import car detection and lane detection modules to combine results
import car_detection
import lane_detection

import cv2
import matplotlib.pyplot as plt

# Import needed to work with video clips
from moviepy.editor import VideoFileClip


def process_image(image):
    '''Function used to process video clips with our lane identification'''

    # Detect lanes
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result, undist = lane_detection.identify_lanes_curvature_center(image,
                    lane_detection.cameraMatrix, lane_detection.distCoeffs,
                    lane_detection.M, lane_detection.Minv)
    
    # Detect cars
    car_detection.car_boxes.update_boxes(undist)
    # Extract bounding boxes considering previous frames
    bboxes = car_detection.car_boxes.extract_accurate_boxes()
    # Draw boxes on the result image
    if bboxes:
         result = car_detection.draw_boxes(result, bboxes)
    result = car_detection.convert_color(result, 'RGB')
    # plt.imshow(result)
    # plt.pause(0.1)
    return result


if __name__ == "__main__":

    clip = VideoFileClip('project_video.mp4')
    output = 'output_images/project_video_processed.mp4'
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output, audio=False)