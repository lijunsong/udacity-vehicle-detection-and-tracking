import matplotlib.image as mpimg
import numpy as np
import cv2
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from glob import glob
from utils import (extract_features, slide_window, draw_boxes, find_cars)

import os
import pickle

debug = True

def log(msg):
    if debug:
        print("[debug]", msg)

def read_image(path):
    """read image in RGB"""
    assert(os.path.exists(path))
    return mpimg.imread(path)

class CarImages(object):
    def __init__(self, car_folder, noncar_folder):
        log("Looking for image files ...")
        self.car_paths = self.get_files(car_folder)
        self.noncar_paths = self.get_files(noncar_folder)

        self.car_images = [read_image(i) for i in self.car_paths]
        self.noncar_images = [read_image(i) for i in self.noncar_paths]


    def get_files(self, folder, suffix_list=['.jpeg', '.png', '.jpg']):
        """list all files with the given suffix"""
        assert(os.path.exists(folder))
        res = []
        for root, dirs, files in os.walk(folder):
            res += [os.path.join(root, f) for f in files if os.path.splitext(f)[1] in suffix_list]
        return res

    def get_all_image(self):
        """Get a list of images and their labels"""
        X = np.vstack((self.car_images, self.noncar_images))
        y = np.hstack((np.ones(len(self.car_images)),
                       np.zeros(len(self.noncar_images))))
        return X, y

color_space = 'YCrCb'

class Model(object):
    """Given Training features and labels, When training, this class will
    split data into train and test set before training

    """
    def __init__(self, model_file=None):
        if model_file is None:
            self.model = LinearSVC()
            self.scalar = None
        else:
            with open(model_file, 'rb') as f:
                self.model, self.scalar = pickle.load(f)

    def save_model(self):
        with open('model.pkl', 'wb') as f:
            pickle.dump((self.model,self.scalar), f)

    def preprocess(self, images):
        "preprocess a list of images to produce their features"
        log("Preprocess {} images".format(len(images)))
        features = []
        for i in images:
            features.append(extract_features(i, color_space=color_space, hog_channel='ALL'))
        print("feature length", len(features[0]), "feature shape", features[0].shape)
        return np.array(features).astype(np.float64)

    def train(self, X, y, test_size=0.3):
        """Given a list of images and their label, train the model."""
        features = self.preprocess(X)

        self.scalar = StandardScaler().fit(features)
        scaled_X = self.scalar.transform(features)

        stat = np.random.randint(0, 100)
        scaled_X, y = shuffle(scaled_X, y)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,
                                                            test_size=test_size,
                                                            random_state=stat)
        log("Training...")
        self.model.fit(X_train, y_train)

        score = self.model.score(X_test, y_test)

        log("Testing score: {}".format(score))

    def predict(self, feature):
        """Given a feature, predict whether this is a vehicle.
        Note: feature is not an image"""
        return self.model.predict(feature)

class CarSearch(object):
    def __init__(self, model):
        self.model = model
        # record history of heat (a copy)
        self.history = collections.deque(maxlen=7)

    def __search_cars_in_image(self, img):
        image_size = img.shape
        box_list = []
        def search(ybeg, yend, scale):
            return find_cars(img, color_space, ybeg, yend, scale,
                             self.model, self.model.scalar,
                             show_all=False)
        #box_list += search(390, 650, 2.5)
        #box_list += search(390, 650, 1.8)

        box_list += search(370, 500, 1)
        box_list += search(400, 580, 2)
        #box_list += search(400, 580, 2.5)

        return box_list

    def annotate_cars_in_image(self, img):
        box_list = self.__search_cars_in_image(img)
        cars = draw_boxes(img, box_list)
        return cars

    def __add_heat(self, box_list):
        """add heat to the history"""
        heat = np.zeros((720, 1280)) #hardcode for now
        for box in box_list:
            heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        self.history.append(heat)

    def generate_heat(self):
        heat = np.sum(np.array(self.history), axis=0)
        if len(self.history) == self.history.maxlen:
            # should stay at least 4 frames
            heat[heat <= 3] = 0
        return heat

    def annotate_cars_in_video(self, img):
        box_list = self.__search_cars_in_image(img)
        self.__add_heat(box_list)

        heat = self.generate_heat()

        pixels, ncars = label(heat)

        new_boxes = []
        for i in range(1, ncars+1):
            nonzero = (pixels == i).nonzero()

            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            new_boxes.append(box)

        draw_boxes(img, new_boxes, make_copy=False)
        # imsave take cmap='hot' to save the heatmap
        #return np.hstack((img, hot))
        return img

def training(vehicle_folder, nonvehicle_folder):
    training_data = CarImages(vehicle_folder, nonvehicle_folder)
    model = Model()
    search = CarSearch(model)

    X, y = training_data.get_all_image()
    model.train(X, y)
    model.save_model()
    return model

if __name__ == '__main__':
    training('large-data/vehicles', 'large-data/non-vehicles')
