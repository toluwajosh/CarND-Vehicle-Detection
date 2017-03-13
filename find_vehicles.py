# Import Needed Libraries
import os
import cv2
import time
import itertools
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

from Features import get_hog_features, bin_spatial, color_hist, single_img_features, slide_window, extract_features

np.random.seed(seed=1987)

############################################################################
# %% Load Data
vehicles_path = 'vehicles/vehicles'
non_vehicles_path = 'non-vehicles/non-vehicles'


vehicles = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(vehicles_path)
    for f in files if f.endswith('.png')]

non_vehicles = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(non_vehicles_path)
    for f in files if f.endswith('.png')]

print ('\nNumber of Vehicles dataset:',len(vehicles))
print ('Number of Non Vehicles dataset:',len(non_vehicles),'\n')

#############################################################################
# %% Train Classifier
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 64    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [480, None] # Min and max in y to search in slide_window()

t=time.time()

vehicle_features = extract_features(vehicles, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
non_vehicle_features = extract_features(non_vehicles, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')

# Create an array stack of feature vectors
X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)

# Normalize features: by removing the mean and scaling to unit variance
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

####################################################################################
# %% class for vehicle detection in video frames
class Vehicle(object):
    """Class for Vehicle detection in a video frame"""
    def __init__(self):
        # Initialize parameters
        self.y_start_stop = [400, 656] # Min and max in y to search in slide_window()
        self.ystart = 400
        self.ystop = 656
        self.scales = [1.0, 1.5]
        self.sliding_win_sizes = [96] # ,128
        self.overlap = 0.8
        self.spatial_size = (32, 32)
        self.hist_bins = 64
        self.orient = 8
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.color_space = 'YCrCb'
        self.hog_channel = 'ALL'
        self.hog_feat = True
        self.spatial_feat = True
        self.hist_feat = True

        # to keep track of heat within consecutive frames
        self.recent_heats = []
        self.ave_heat = []

    def update_heat(self, current_heat, patience=16):
        """
        Update average heat in frames using n past frames
        """
        self.recent_heats.append(current_heat)
        if len(self.recent_heats) > patience:
            self.recent_heats.pop(0)

        self.ave_heat = np.intc(np.average(self.recent_heats, 0))


    def search_windows(self, img, windows, clf, scaler, color_space='YCrCb',
                    spatial_size=(32, 32), hist_bins=64,
                    hist_range=(0, 256), orient=8,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel='ALL', spatial_feat=True,
                    hist_feat=True, hog_feat=True):
        """
        Returns windows of positive detection from a list of windows to be searched
        """

        # An empty list to receive positive detection windows
        on_windows = []
        # Iterate over all windows in the list
        for window in windows:
            # Extract the test window from original image
            # print window
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # Extract features for that window using single_img_features()
            features = single_img_features(test_img, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
            # Scale extracted features to be fed to classifier
            # Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            # Predict using your classifier
            prediction = clf.predict(test_features)
            # If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        # Return windows for positive detections
        return on_windows

    def sliding_window(self, frame):

        windows = []
        for size in self.sliding_win_sizes:
            windows.append(slide_window(frame, x_start_stop=[None, None], y_start_stop=self.y_start_stop,
                                    xy_window=(size,size), xy_overlap=(self.overlap, self.overlap)))
        windows = list(itertools.chain(*windows))

        frame = frame.astype(np.float32)/255

        bboxes = self.search_windows(frame, windows, svc, X_scaler, color_space=self.color_space,
                            spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                            orient=self.orient, pix_per_cell=self.pix_per_cell,
                            cell_per_block=self.cell_per_block, hog_channel=self.hog_channel,
                            spatial_feat=self.spatial_feat, hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        return bboxes


    def find_cars(self, img, ystart, ystop, scale, svc, X_scaler, orient,
                    pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space='YCrCb'):
        """
        Extract features using hog subsampling and makde predictions
        """
        # draw_img = np.copy(img)
        img = img.astype(np.float32)/255

        img_tosearch = img[ystart:ystop,:,:]
        # ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
        else: ctrans_tosearch = np.copy(img_tosearch)

        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1
        # nfeat_per_block = orient*cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        bboxes = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)


                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                # test_features = X_scaler.transform(np.hstack((hist_features, hog_features)).reshape(1, -1))
                test_prediction = svc.predict(test_features)
                #test_prediction = svc.predict(hog_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    bboxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                    # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

        return bboxes


    def hog_subsample(self, frame):
        bboxes = []
        for scale_val in self.scales:
            boxes_list = self.find_cars(frame, ystart=self.y_start_stop[0], ystop=self.y_start_stop[1],
                                    scale=scale_val, svc=svc, X_scaler=X_scaler, orient=self.orient,
                                    pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                    spatial_size=self.spatial_size, hist_bins=self.hist_bins)
            bboxes.append(boxes_list)
        bboxes = list(itertools.chain(*bboxes))
        return bboxes

    # %% for computing heat maps
    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap# Iterate through list of bboxes

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img


    def vehicle_find(self, frame):
        """
        Runs the vehicle detect pipeline on a frame
        """
        # using sliding window ################
        # bboxes = self.sliding_window(frame)
        # #####################################

        # using HOG subsampling: much faster implementation of sliding window
        bboxes = self.hog_subsample(frame)
        # #####################################

        heat = np.zeros_like(frame[:,:,0]).astype(np.float)

        # Add heat to each box in box list
        heat = self.add_heat(heat,bboxes)

        self.update_heat(heat)

        # Apply threshold to help remove false positives
        heat = self.apply_threshold(self.ave_heat, 2)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = self.draw_labeled_bboxes(np.copy(frame), labels)

        return draw_img


# %% View solution in real time
vehicle_frames = Vehicle() # initialize

name_prefix = 'test'
vid_path = name_prefix+'_video.mp4'

# %% OpenCV:
# open video cap
cap = cv2.VideoCapture(vid_path)

print(cap.isOpened())

frame_no = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # cv2.imwrite(vid_name+'/'+str(frame_no)+'.jpg', frame)

        draw_img = vehicle_frames.vehicle_find(frame=frame)

        cv2.imshow('frame',draw_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else: break
cap.release()
cv2.destroyAllWindows()

# %% Export finished video
from moviepy.editor import VideoFileClip
vid_output = name_prefix+'_solve_3.mp4'
clip_source = VideoFileClip(vid_path)
vid_clip = clip_source.fl_image(vehicle_frames.vehicle_find) #NOTE: this function expects color images!!
vid_clip.write_videofile(vid_output, audio=False)