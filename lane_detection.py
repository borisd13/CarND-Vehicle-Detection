import numpy as np
import cv2
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Import needed to work with video clips
from moviepy.editor import VideoFileClip



# Define a class to receive the characteristics of each line detection
class Line():

    def __init__(self):
        # how many frames ago was line last detected
        self.last_detected = float('inf')
        self.DETECTION_THRESHOLD = 10   # max number of frames with no detection acceptable
        # prev fits of lanes
        self.left_fit, self.right_fit = None, None
        self.curv = None
        self.dist_center = None

    def update(self, left_fit, right_fit):
        '''Update all the parameters of Line with some consistency check'''

        # This is the fist time we add values
        if self.left_fit is None:
            self.left_fit = left_fit
            self.right_fit = right_fit
            self.last_detected = 0

        # We have not detected good fits for a while
        elif self.last_detected > self.DETECTION_THRESHOLD:

            # we check that lines are parallel based on 10 points
            ploty = np.linspace(0, 720-1, 10)
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            diff = left_fitx - right_fitx
            avg_distance = np.average(diff)
            parallelism = np.sum(np.abs(diff - avg_distance))
            if parallelism < 500:  # threshold defined experimentally
                self.left_fit = left_fit
                self.right_fit = right_fit
                self.last_detected = 0

        else:
            # we evaluate the difference between last fit and current fit on 10 points
            ploty = np.linspace(0, 720-1, 10)
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            prev_left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            prev_right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

            # thresholds to define quality of the fits have been defined experimentally
            left_diff = np.sum(np.abs(left_fitx - prev_left_fitx))
            right_diff = np.sum(np.abs(right_fitx - prev_right_fitx))
            diff = left_fitx - right_fitx
            avg_distance = np.average(diff)
            parallelism = np.sum(np.abs(diff - avg_distance))
            if ((left_diff < 200) and (right_diff < 200)) or parallelism < 300:
                coeff = 1
                self.last_detected = 0
            elif ((left_diff < 350) and (right_diff < 350)) or parallelism < 400:
                coeff = 0.9
                self.last_detected = 0
            elif ((left_diff < 600) and (right_diff < 600)) or parallelism < 500:
                coeff = 0.8
                self.last_detected = 0
            elif ((left_diff < 800) and (right_diff < 800)) or parallelism < 800:
                coeff = 0.3
                self.last_detected = 0
            else: # values seem to be wrong
                coeff = 0
                self.last_detected += 1
            # we perform an exponential moving average based on the coefficient we defined
            self.left_fit = coeff * left_fit + (1 - coeff) * self.left_fit
            self.right_fit = coeff * right_fit + (1 - coeff) * self.right_fit

        self.curv, self.dist_center = calculate_curvature_and_center(self.left_fit, self.right_fit)
        

    def get_fits(self):
        # get parameters defining each line
        if self.last_detected < self.DETECTION_THRESHOLD:
            return self.left_fit, self.right_fit
        return None, None

    def get_all(self):
        # get all parameters
        return self.left_fit, self.right_fit, self.curv, self.dist_center


# Define a Line object as global
line = Line()


def calibrate_camera(chessboard_folder = 'camera_cal'):
    '''Calibrate the camera.BaseException

    Open the picture of a chessboard and use it to create the calibration matrix of the camera.

    Args:
        chessboard_path: Path of a picture of a chessboard used for calibration.
        
    Returns:
        cameraMatrix, distCoeffs: camera matrix and distortion coefficients'''

    # Number of intersections of the chessboard on the picture
    nx, ny = 9, 6

    # Define chessboard real coordinates projected to plane z=0
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Go through all the pictures and add chessboard corners
    for image_path in images:

        # Read the picture and convert it to gray
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret == True:

            # Add the coordinates to the specified lists
            objpoints.append(objp)
            imgpoints.append(corners)

    # Calculate the calibration parameters (camera matrix and distortion coefficients)
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return cameraMatrix, distCoeffs


def undistort(image, cameraMatrix, distCoeffs, display = False):
    '''Undistort a picture and save a copy in output_images

    Args:
        image: image to undistort
        cameraMatrix: camera matrix as returned by cv2.calibrateCamera
        distCoeffs: distribution coefficients as returned by cv2.calibrateCamera
        display: display results of transformation'''

    # Read the image
    undist = cv2.undistort(image, cameraMatrix, distCoeffs)

    # Display pictures
    if display:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
        ax2.set_title('Undistorted Image', fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return undist


def create_binary(image, display = False):
    '''Create a binary picture to identify the lanes.

    Args:
        image: an image that has already been undistorted
        display: display results of transformation'''

    # Create a filter based on the saturation channel of HLS format
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_thresh = 90, 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Create a filter based on sobel in x direction on red channel
    r_channel = image[:, :, 2].astype(np.float)
    rsobel_thresh = 30, 100
    sobel_abs = np.absolute(cv2.Sobel(r_channel, cv2.CV_64F, 1, 0))
    sobel_abs = np.uint8(255 * sobel_abs / np.max(sobel_abs))
    rsobel_binary = np.zeros_like(r_channel)
    rsobel_binary[(sobel_abs > rsobel_thresh[0]) & (sobel_abs <= rsobel_thresh[1])] = 1

    # Combine both filters
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1) | (rsobel_binary == 1)] = 1

    # Display Filters
    if display:
        color_binary = np.dstack((np.zeros_like(s_binary), s_binary, rsobel_binary))
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
        ax1.set_title('Original')
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax2.set_title('HLS Saturation')
        ax2.imshow(s_binary, cmap='gray')
        ax3.set_title('Sobel on Red Channel')
        ax3.imshow(rsobel_binary, cmap='gray')
        ax4.set_title('Combined')
        ax4.imshow(color_binary, cmap='gray')
        plt.show()

    return combined_binary


def create_transform_matrix():
    '''Create a tranform matrix and its inverse to switch between front view and top view'''

    # Standard sizes of the pictures
    size_x, size_y = 1280, 720

    # We defined straight points from the road on picture straight_lines1.jpg
    front_view_corners = np.float32([[248, 690], [1057, 690], [611, 440], [667, 440]])

    # We define desired projections
    top_view_corners = np.float32([[350, size_y], [size_x - 350, size_y],
                                   [350, 0], [size_x - 350, 0]])

    # We create the transform matrixes
    M = cv2.getPerspectiveTransform(front_view_corners, top_view_corners)
    Minv = cv2.getPerspectiveTransform(top_view_corners, front_view_corners)

    return M, Minv


def create_transform(image, Matrix_transform, display = False):
    '''Project image using a transformation matrix.

    Args:
        image: picture to project
        Matrix_transform: transformation matrix used to transform the picture
        display: display results of transformation'''

    warped = cv2.warpPerspective(image, Matrix_transform, (image.shape[1], image.shape[0]))

    if display:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Original')
        ax2.set_title('Projection')
        if len(image.shape) == 3:            
            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))            
            ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        else:
            ax1.imshow(image, cmap='gray')
            ax2.imshow(warped, cmap='gray')
        plt.show()

    return warped


def define_polynom_and_curvatures(image, display = False):
    '''Find the lanes on a projected binary picture.

    Args:
        image: projected binary picture'''

    def window_masked(width, height, img_ref, center, level):
        '''Helper function to return an image masked by a region'''

        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
               max(0, int(center-width/2)):min(int(center+width/2), img_ref.shape[1])] = img_ref[
                   int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
                   max(0, int(center-width/2)):min(int(center+width/2), img_ref.shape[1])]
        return output


    def find_lane_points_no_basis(image):
        '''Find left and right points when with no previous reference'''

        # Define standard parameters for images used
        window_width = 50
        window_height = 80 # Break image into 9 vertical layers since image height is 720
        margin = 100 # How much to slide left and right for searching

        # First find the two starting positions for the left and right lane by using np.sum to get the
        # vertical image slice and then np.convolve the vertical image slice with the window template

        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions

        # Sum 1/8 bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(7*image.shape[0]/8):, :int(image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum))-window_width/2
        r_sum = np.sum(image[int(7*image.shape[0]/8):, int(image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum))-window_width/2+int(image.shape[1]/2)

        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(image.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height), :], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin, 0))
            l_max_index = int(min(l_center+offset+margin, image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin, 0))
            r_max_index = int(min(r_center+offset+margin, image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center, r_center))

        # Get left and right points coordinates
        l_points_x, l_points_y = [], []
        r_points_x, r_points_y = [], []
        for level in range(0,len(window_centroids)):
            # mask the image on previously detected windows
            l_mask = window_masked(window_width, window_height, image, window_centroids[level][0], level)
            r_mask = window_masked(window_width, window_height, image, window_centroids[level][1], level)
            # Add graphic points from window mask here to each side
            l_non_zero = l_mask.nonzero()
            r_non_zero = r_mask.nonzero()
            l_points_x.append(l_non_zero[1])
            l_points_y.append(l_non_zero[0])
            r_points_x.append(r_non_zero[1])
            r_points_y.append(r_non_zero[0])
        l_points_x = np.concatenate(l_points_x)
        l_points_y = np.concatenate(l_points_y)
        r_points_x = np.concatenate(r_points_x)
        r_points_y = np.concatenate(r_points_y)

        return l_points_x, l_points_y, r_points_x, r_points_y


    def find_lane_points_previous_reference(image, prev_left_fit, prev_right_fit):
        '''Find left and right points when with the help of previous reference'''

        # Define standard parameters for images used
        window_width = 20
        window_height = 30 # Break image into small sections because we know where to look for the lane
        
        # Use previous fits of lanes to look for points within a small window on each row
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + prev_left_fit[2] - window_width/2)) & (nonzerox < (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + prev_left_fit[2] + window_width/2))) 
        right_lane_inds = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + prev_right_fit[2] - window_width/2)) & (nonzerox < (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + prev_right_fit[2] + window_width/2)))  

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty


    # Check if we have previous fits we can use
    previous_left_fit, previous_right_fit = line.get_fits()
    if previous_left_fit is not None:
        leftx, lefty, rightx, righty = find_lane_points_previous_reference(image, previous_left_fit, previous_right_fit)
    else:
        leftx, lefty, rightx, righty = find_lane_points_no_basis(image)

    # Fit polynoms on each side
    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 150 else np.array([0., 0., 0.])
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 150 else np.array([100., 100., 100.]) # to ensure they are not parallel
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])  # To display results and evaluate curvature and center

    # Display results
    if display:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        out_img = np.dstack((image, image, image))*255
        out_img[lefty, leftx] = (255, 0, 0)
        out_img[righty, rightx] = (0, 255, 0)
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    # Update our global Line object to keep track of measurements
    line.update(left_fit, right_fit)

    # Get updated and calculated parameters
    left_fit, right_fit, curv, dist_center = line.get_all()

    return (left_fit, right_fit, curv, dist_center)


def calculate_curvature_and_center(left_fit, right_fit):
    '''Function to calculate the curvature of 2 lanes based on their polynom representation

    Args:
        left_fit: polynom defining left lane
        right_fit: polynom defining right lane

    Returns:
        tuple of curvature and distance from center in meters'''

    # Define conversions in x and y from pixels space to meters, measured on projection of straight_lines2.jpg
    ym_per_pix = 30/500 # meters per pixel in y dimension
    xm_per_pix = 3.7/600 # meters per pixel in x dimension

    # Define a few parameters we will need
    y_eval = 720 # Location where we calculate curvature
    ploty = np.linspace(0, y_eval-1, 5)  # 5 points are more than enough to define a 2nd degree polynom

    # Define a center line between left and right lanes
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    center_fitx = (left_fitx + right_fitx) / 2.

    # Fit new polynomials to x,y in world space using the center line
    center_fit = np.polyfit(ploty, center_fitx, 2)
    center_fit_cr = np.polyfit(ploty*ym_per_pix, center_fitx*xm_per_pix, 2)

    # Calculate the radius of curvature
    curv = ((1 + (2*center_fit_cr[0]*y_eval*ym_per_pix + center_fit_cr[1])**2)**1.5) / np.absolute(2*center_fit_cr[0])

    # Calculate distance from center
    center_fit_basis = center_fit[0]*y_eval**2 + center_fit[1]*y_eval + center_fit[2]
    dist_from_center = (center_fit_basis - 1280/2) * xm_per_pix

    # Now our radius of curvature is in meters
    return curv, dist_from_center


def identify_lanes_curvature_center(image, cameraMatrix, distCoeffs, M, Minv, display=False):
    '''Display lanes, curvature and center information on an image'''

    # First we undistort the picture
    undist = undistort(image, cameraMatrix, distCoeffs)

    # Then we create a binary picture that selects the lanes
    binary = create_binary(undist)

    # We project it in a top view
    warped = create_transform(binary, M)

    # We identify the lanes through a polynom and evaluate the curvatures and distance from center
    left_fit, right_fit, curv, dist_center = define_polynom_and_curvatures(warped)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    cv2.putText(result, 'Curvature: {:.0f}m'.format(curv), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
    cv2.putText(result, 'Distance from center: {:.2f}m'.format(dist_center), (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)

    if display:
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()

    return result, undist


def process_image(image):
    '''Function used to process video clips with our lane identification'''

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return cv2.cvtColor(identify_lanes_curvature_center(image, cameraMatrix, distCoeffs, M, Minv)[0], cv2.COLOR_BGR2RGB)


# Create necessary global variables

# Calibrate the camera
cameraMatrix, distCoeffs = calibrate_camera()

# Create the transform matrix and its inverse
M, Minv = create_transform_matrix()


if __name__ == "__main__":

    # Optional: Undistort a test picture
    # img = cv2.imread('camera_cal/calibration1.jpg')
    # undistort(img, cameraMatrix, distCoeffs, display=True)

    # Optional: Create a binary image on a test picture
    # img = cv2.imread('test_images/test2.jpg')
    # undist = undistort(img, cameraMatrix, distCoeffs)
    # create_binary(undist, display=True)

    # Optional: Test the transform
    # img = cv2.imread('test_images/straight_lines2.jpg')
    # undist = cv2.undistort(img, cameraMatrix, distCoeffs)
    # create_transform(img, M, display=True)

    # Optional: Test the polynom definition
    # img = cv2.imread('test_images/test3.jpg')
    # undist = undistort(img, cameraMatrix, distCoeffs)
    # binary = create_binary(undist)
    # warped = create_transform(binary, M)
    # _, _, curv, dist_center = define_polynom_and_curvatures(warped, display = True)
    # print('Curvature of {:.0f}m on left side, {:.0f}m on right side, car at {:.2f}m from center'.format(curv, dist_center))

    # Optional: Test the lane and curvature identification on a front view picture
    # image = cv2.imread('test_images/test3.jpg')
    # identify_lanes_curvature_center(image, cameraMatrix, distCoeffs, M, Minv, display=True)

    # Optional: process a video
    clip = VideoFileClip('project_video.mp4')
    output = 'output_images/project_video_processed.mp4'
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output, audio=False)
