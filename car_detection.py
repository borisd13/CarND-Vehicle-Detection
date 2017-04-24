import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
import glob
import pickle
import pathlib

# Debug
import logging
logging.basicConfig(filename='info.log', filemode='w', level=logging.DEBUG)

# Import needed to work with video clips
from moviepy.editor import VideoFileClip


def draw_boxes(img, bboxes, color=(255, 0, 0), thick=5):
    # make a copy of the image
    draw_img = np.copy(img)
    for c1, c2 in bboxes:
        cv2.rectangle(draw_img, c1, c2, color, thick)

    return draw_img # Change this line to return image copy with boxes


def convert_color(img, color_space='BGR'):
    '''Convert an image to another color space.

    Args:
        img: an image represented in BGR (through cv2.imread)
        color_space: can be BGR, HSV, LUV, HLS, YUV, YcrCb

    Returns:
        converted image'''

    if color_space != 'BGR':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        elif color_space == 'RGB':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        feature_image = np.copy(img)

    return feature_image


def color_hist(img, nbins=32, bins_range=(0, 256)):
    '''Create features based on the histogram of an image

    Args:
        img: an image in any color space
        nbins: number of bins to create the histogram
        bins_range: range of the pixel values

    Returns:
        feature vector based on the histogram of each channel'''

    # Compute the histogram of the 3 channels separately
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

    return hist_features


def bin_spatial(img, size=(16, 16)):
    '''Creates features based on pixel intensity of a resized image

    Args:
        img: an image represented in any color space
        size: format to use for resizing the image

    Returns:
        features based on the concatenation of pixels from resized image'''

    features = cv2.resize(img, size).ravel()

    return features


def get_training_data(display=False):
    '''Get pictures of cars and non-cars

    Args:
        display: shows example of data

    Returns:
        dict object with "cars" and "non-cars" keys returning list of files,
        and shape and dtype keys to define the pictures.'''

    # initialize list of files
    cars = []
    notcars = []

    for path in glob.glob('./training/**/*.png', recursive=True):
        if 'non-vehicles' in path:
            notcars.append(path)
        else:
            cars.append(path)

    tmp = cv2.imread(cars[0])

    if display:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        car, non_car = cv2.imread(random.choice(cars)), cv2.imread(random.choice(notcars))
        ax1.imshow(cv2.cvtColor(car, cv2.COLOR_BGR2RGB))
        ax1.set_title('Car', fontsize=20)
        ax2.imshow(cv2.cvtColor(non_car, cv2.COLOR_BGR2RGB))
        ax2.set_title('Not car', fontsize=20)
        plt.show()
        return car, non_car

    return {'cars': cars, 'non-cars': notcars, 'shape': tmp.shape, 'type': tmp.dtype}


def get_hog_features(img_gray, orient, pix_per_cell, cell_per_block, visual=False, feature_vec=True):
    '''Returns the HOG features of an image

    Args:
        img_gray: image in gray scale
        orient: number of orientation bins
        pix_per_cell: number of pixels in each cell
        cell_per_block: number of cells in each block
        visual: boolean for creation of a visualisation
        feature_vec: results returned as a vector

    Returns:
        HOG features of an image'''

    if visual == True:
        features, hog_image = hog(img_gray, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), visualise=visual,
                                  feature_vector=feature_vec)
        return features, hog_image

    else:
        features = hog(img_gray, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), visualise=visual,
                       feature_vector=feature_vec)

        return features


def extract_features(imgs, color_space='BGR', spatial_size=16, hist_bins=32, hist_bin_range=(0, 256),
                     hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2, hog_channels=range(3)):
    '''Extract features of an image list by combining HOG, histogram of pixels and spatial pixel data'''

    features = []

    for img in imgs:

        # Load and onvert the image to desired color space
        img_conv = convert_color(cv2.imread(img), color_space)

        # Extract all features of interest and combine them
        features_spatial = bin_spatial(img_conv, size=(spatial_size, spatial_size))
        features_hist = color_hist(img_conv, nbins=hist_bins, bins_range=hist_bin_range)
        features_hog = []
        for channel in hog_channels:
            features_hog.append(get_hog_features(img_conv[:, :, channel], hog_orient, hog_pix_per_cell,
                                                 hog_cell_per_block, visual=False, feature_vec=True))
        features_hog = np.ravel(features_hog)

        # Concatenate all features and append to the list
        features.append(np.concatenate([features_spatial, features_hist, features_hog]))

    return features


def train_car_data_set(color_space='YCrCb', spatial_size=16, hist_bins=32, hist_bin_range=(0, 256),
                       hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2, hog_channels=range(3)):
    '''Train data set to recognize car pictures'''

    # Load data
    data = get_training_data()

    # Extract features on all pictures
    feat_cars = np.array(extract_features(data['cars'], color_space, spatial_size, hist_bins,
                    hist_bin_range, hog_orient, hog_pix_per_cell, hog_cell_per_block, hog_channels), dtype=np.float64)
    feat_non_cars = np.array(extract_features(data['non-cars'], color_space, spatial_size, hist_bins,
                        hist_bin_range, hog_orient, hog_pix_per_cell, hog_cell_per_block, hog_channels), dtype=np.float64)

    # Stack the features and normalize them
    X = np.vstack((feat_cars, feat_non_cars))
    X_scaler = StandardScaler().fit(X)
    X = X_scaler.transform(X)

    # Create labels vector
    y = np.hstack((np.ones(len(feat_cars)), np.zeros(len(feat_non_cars))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Use a linear SVC 
    svc = LinearSVC()
    svc.fit(X_train, y_train)

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # return the SVC and normalizer
    return svc, X_scaler


def find_cars(img, svc, X_scaler, scales = [1., 2.], ystart=360, ystop=670, color_space='YCrCb', spatial_size=16, hist_bins=32,
              hist_bin_range=(0, 256), hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2, hog_channels=range(3)):
    '''Find cars in a picture based on a pre-trained classifier

    Args:
        ystart, ystop: vertical range of pixels to consider
        scale: scale used on the classifier
        svc, X_scaler: pre-trained classifier and its data normalizer
        remaining parameters must be the same as used on classifier

    Returns:
        list of bounding boxes to define car position'''

    # Create a list of bounding boxes to identify cars
    car_list = []

    # Identify cars on various scales
    for scale in scales:

        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = convert_color(img_tosearch, color_space)

        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ctrans_tosearch.shape[1] // hog_pix_per_cell)-1
        nyblocks = (ctrans_tosearch.shape[0] // hog_pix_per_cell)-1
        nfeat_per_block = hog_orient*hog_cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // hog_pix_per_cell)-1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        features_hog = dict()
        for channel in hog_channels:
            features_hog[channel] = get_hog_features(ctrans_tosearch[:, :, channel], hog_orient, hog_pix_per_cell,
                                                    hog_cell_per_block, visual=False, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                patch_hog_feat = []
                for channel in hog_channels:
                    patch_hog_feat.append(features_hog[channel][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel())

                patch_hog_feat = np.ravel(patch_hog_feat)

                xleft = xpos*hog_pix_per_cell
                ytop = ypos*hog_pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                features_spatial = bin_spatial(subimg, size=(spatial_size, spatial_size))
                features_hist = color_hist(subimg, nbins=hist_bins, bins_range=hist_bin_range)

                # Concatenate all features and append to the list
                features = np.hstack((features_spatial, features_hist, patch_hog_feat)).reshape(1, -1)

                # Scale features and make a prediction
                features = X_scaler.transform(features)
                prediction = svc.predict(features)

                if prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    car_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

    return car_list


def create_heatmap(bbox_list, shape):
    '''Create a heatmap based on a list of boxes

    Args:
        bbox_list: list of boxes in the form ((x1, y1), (x2, y2))
        shape: shape of the heatmap (size_1, size_2)

    Returns:
        heatmap counting number of boxes a pixel is in'''

    heatmap = np.zeros(shape)
    x_margin, y_margin = 10, 5  # margin around detected picture to include edges

    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]-y_margin:box[1][1]+y_margin, box[0][0]-x_margin:box[1][0]+x_margin] += 1

    return heatmap


def apply_threshold(heatmap, threshold, binary=False):
    '''Zero out pixels below a certain value.

    Convert positive pixels to 1 if binary flag is True'''

    heatmap[heatmap < threshold] = 0
    if binary:
        heatmap[heatmap >= threshold] = 1
    return heatmap


def define_potential_pixels(bbox_list, shape, threshold):
    '''Create potential positions of boxes based on a threshold

    A heatmap is first created and a threshold is applied for false positives.
    The position of potential pixels is then returned.

    Args:
        bbox_list: list of boxes in the form ((x1, y1), (x2, y2))
        shape: shape of the heatmap (size_1, size_2)
        threshold: minimum number of boxes a pixel need to belong to

    Returns:
        Position of pixels potentially belonging to an identified box'''

    heatmap = create_heatmap(bbox_list, shape)
    apply_threshold(heatmap, threshold, binary=True)
    return heatmap


def define_accurate_position(heatmap_list, shape, threshold):
    '''Create accurate boxes based on a list of potential pixels

    A heatmap is first created and a threshold is applied for false positives.

    Args:
        bbox_list: list of boxes in the form ((x1, y1), (x2, y2))
        shape: shape of the heatmap (size_1, size_2)
        threshold: minimum number of boxes a pixel need to belong to

    Returns:
        list of accurate position of boxes'''

    accurate_bbox_list = []
    min_size_box_x, min_size_box_y = 70, 70   # min size of a detected box to avoid noise
    heatmap = sum(heatmap_list)
    apply_threshold(heatmap, threshold)
    labels = label(heatmap)

    # define a box for each detected region
    logging.info(' New frame '.center(50, '*'))
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = (np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))
        # check that the detected box is large enough and not just likely an error
        if (np.max(nonzerox) - np.min(nonzerox) > min_size_box_x) and (
                np.max(nonzeroy) - np.min(nonzeroy) > min_size_box_y):
            accurate_bbox_list.append(bbox)
            logging.info('Box :{}, size: {}, {}'.format(bbox, np.max(nonzerox) - np.min(nonzerox), np.max(nonzeroy) - np.min(nonzeroy)))

    return accurate_bbox_list


def process_image(image):
    '''Function used to process video clips with our car detection'''

    # Convert to BGR
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Update the list of boxes based on the new frame
    car_boxes.update_boxes(frame)

    # Extract bounding boxes considering previous frames
    bboxes = car_boxes.extract_accurate_boxes()

    # Draw boxes on the original image
    if bboxes:
        frame = draw_boxes(frame, bboxes)
    frame = convert_color(frame, 'RGB')

    # plt.imshow(frame)
    # plt.pause(0.1)
    return frame


class Car_boxes():
    '''Class to manage position of cars based on successive frames'''

    def __init__(self):

        # Load or train svc if not pre-existing
        svc_path = pathlib.Path('svc_pickle.p')
        if svc_path.exists():
            with svc_path.open('rb') as f:
                svc_data = pickle.load(f)
                svc, scaler = svc_data['svc'], svc_data['scaler']
        else:
            svc, scaler = train_car_data_set()
            # Save trained svc
            with svc_path.open('wb') as f:
                pickle.dump({'svc':svc, 'scaler':scaler}, f)
        self.svc, self.scaler = svc, scaler

        self.boxes = []
        self.max_frames = 30  # number of successive frames to consider for car detection

    def update_boxes(self, frame):
        '''Update list of boxes based on a new frame'''

        if not self.boxes:
            # we need to initialize shape attribute
            self.shape = frame.shape[0], frame.shape[1]

        # if we are at the max of records, remove the oldest
        if len(self.boxes) >= self.max_frames:
            self.boxes.pop(0)

        # add the bounding boxes of the new frame
        self.boxes.append(find_cars(frame, self.svc, self.scaler))

    def extract_accurate_boxes(self):
        '''Extract accurate boxes based on previous frames'''

        all_boxes = []
        for b in self.boxes:
            all_boxes.append(define_potential_pixels(b, self.shape, 2))
        accurate_bboxes = define_accurate_position(all_boxes, self.shape, 0.4 * self.max_frames)
        return accurate_bboxes


# keep track of all car boxes in a global object
car_boxes = Car_boxes()


if __name__ == "__main__":

    # Optional: show training data
    # car, non_car = get_training_data(display=True)
    # car, non_car = convert_color(car, color_space='YCrCb'), convert_color(non_car, color_space='YCrCb')

    # # Optional: show get_hog_features function
    # f, axes = plt.subplots(2, 3)
    # _, hog_image_car = get_hog_features(car[:, :, 0], orient=9, pix_per_cell=8,
    #                                     cell_per_block=2, visual=True)
    # _, hog_image_non_car = get_hog_features(non_car[:, :, 0], orient=9, pix_per_cell=8,
    #                                         cell_per_block=2, visual=True)
    # axes[0][0].imshow(cv2.cvtColor(car, cv2.COLOR_YCrCb2RGB))
    # axes[0][0].set_title('Car', fontsize=10)
    # axes[0][1].imshow(car[:, :, 0], cmap='gray')
    # axes[0][1].set_title('Car, channel 0', fontsize=10)
    # axes[0][2].imshow(hog_image_car, cmap='gray')
    # axes[0][2].set_title('Car, channel 0, HOG', fontsize=10)
    # axes[1][0].imshow(cv2.cvtColor(non_car, cv2.COLOR_YCrCb2RGB))
    # axes[1][0].set_title('Not car', fontsize=10)
    # axes[1][1].imshow(non_car[:, :, 0], cmap='gray')
    # axes[1][1].set_title('Not car, channel 0', fontsize=10)
    # axes[1][2].imshow(hog_image_non_car, cmap='gray')
    # axes[1][2].set_title('Not car, channel 0, HOG', fontsize=10)
    # plt.show()

    # Optional: find cars in a test image and draw bounding boxes
    # test_paths = ['test_images/test1.jpg', 'test_images/test2.jpg', 'test_images/test3.jpg', 'test_images/test4.jpg']
    # _, axes = plt.subplots(2, 2)
    # for k, test_path in enumerate(test_paths):
    #     test_image = cv2.imread(test_path)
    #     car_list = find_cars(test_image, svc, scaler)
    #     img_cars = draw_boxes(test_image, car_list)
    #     axes[k//2, k%2].imshow(convert_color(img_cars, 'RGB'))
    # plt.show()

    # Optional: create heatmap of car detection
    # test_paths = ['test_images/test1.jpg', 'test_images/test2.jpg', 'test_images/test3.jpg', 'test_images/test4.jpg']
    # _, axes = plt.subplots(4, 4)
    # for k, test_path in enumerate(test_paths):
    #     test_image = cv2.imread(test_path)
    #     shape = test_image.shape[0], test_image.shape[1]
    #     car_list = find_cars(test_image, svc, scaler)
    #     img_cars = draw_boxes(test_image, car_list)
    #     axes[k, 0].imshow(convert_color(img_cars, 'RGB'))
    #     heatmap = create_heatmap(car_list, shape)
    #     axes[k, 1].imshow(create_heatmap(car_list, shape), cmap='hot')
    #     axes[k, 2].imshow(apply_threshold(heatmap, 1), cmap='gray')
    #     accurate_bboxes = define_accurate_position(car_list, shape, 1)
    #     img_cars_accurate = draw_boxes(test_image, accurate_bboxes)
    #     axes[k, 3].imshow(convert_color(img_cars_accurate, 'RGB'))
    # plt.show()

    # Optional: process a video
    # clip = VideoFileClip('project_video.mp4').subclip(6, 10)
    # output = 'output_images/project_video_processed_t1.mp4'
    # output_clip = clip.fl_image(process_image)
    # output_clip.write_videofile(output, audio=False)

    # clip = VideoFileClip('project_video.mp4').subclip(20, 30)
    # output = 'output_images/project_video_processed_t2.mp4'
    # output_clip = clip.fl_image(process_image)
    # output_clip.write_videofile(output, audio=False)

    # clip = VideoFileClip('project_video.mp4').subclip(35, 40)
    # output = 'output_images/project_video_processed_t3.mp4'
    # output_clip = clip.fl_image(process_image)
    # output_clip.write_videofile(output, audio=False)

    # clip = VideoFileClip('project_video.mp4')
    # output = 'output_images/project_video_processed.mp4'
    # output_clip = clip.fl_image(process_image)
    # output_clip.write_videofile(output, audio=False)
    pass