#!/usr/bin/env python3
import sys
sys.path.append(".")
from utils import extract_features, ransac, show_images, save_images, create_stitched_image
import os
import cv2
import glob
import fargv  #  pip install fargv or comment line 49 {Laziest command-line argument parser}
import os

parameters = {
    "images_glob": "resources/*jpg",
    "features_per_image": 500,
    "max_iterations": 1000,
    "ransac_threshold": 2.0,
    "significant_descriptor_threshold": .7,
    "ransac_points": 4,
    "reference_name": "4.jpg",
    "rendering_order": ["4.jpg", "3.jpg", "5.jpg", "2.jpg", "6.jpg", "1.jpg", "7.jpg", "0.jpg", "8.jpg"],
    "scale": 0.8,
    "output_file_name": "output.png"
}


def new_main(parameters):
    output_path = os.path.join(os.getcwd(), "results")
    load_img = lambda x: cv2.resize(cv2.imread(x), None, fx=parameters["scale"], fy=parameters["scale"])
    filenames = glob.glob(parameters["images_glob"])  #  No guarantee of order
    images = {os.path.basename(path): load_img(path) for path in filenames}
    features = {path: extract_features(images[path], parameters["features_per_image"]) for path in images.keys()}
    points = {path: features[0] for path, features in features.items()}
    descriptors = {path: features[1] for path, features in features.items()}
    print(len(descriptors))
    names = list(sorted(images.keys()))
    homographies = {}
    for n in range(1, len(names)):
        name1 = names[n-1]
        name2 = names[n]
        homographies[(name1, name2)] = ransac((points[name1], descriptors[name1]),
                                                  (points[name2], descriptors[name2]),
                                                  steps=parameters["max_iterations"], distance_threshold=parameters["ransac_threshold"],
                                                  similarity_threshold=parameters["significant_descriptor_threshold"],
                                                  n_points=parameters["ransac_points"])
    panorama = create_stitched_image(images, homographies, parameters["reference_name"], parameters["rendering_order"])
    show_images([panorama], ["output"])
    save_images([panorama], [os.path.join(output_path, parameters["output_file_name"])])


if __name__ == "__main__":
    # parameters, _ = fargv.fargv(parameters)  #  If you have not installed fargv using pip install fargv: Comment or Remove this line 
    print(parameters['images_glob'])
    new_main(parameters)
