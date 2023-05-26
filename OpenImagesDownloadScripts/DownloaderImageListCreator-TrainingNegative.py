# This script will take the annotations .csv file and create a text file 
# called ImageList.txt for use with the Downloader.py script from
# OpenImages which is available at this URL:
# https://raw.githubusercontent.com/openimages/dataset/master/downloader.py
# This will create the downloader file for the TRAINING set. Modified to 
# take multiple class_label inputs for the purpose of creating a negative
# dataset. It will also specifically exclude the class to be trained.
# Edit class_labels, exclude_class, and num_samples. 

import csv
import random
import os

csv_file = 'oidv6-train-annotations-bbox.csv'

# Change class_labels and exclude_labes based on the class labels of interest
# https://storage.googleapis.com/openimages/2017_07/class-descriptions.csv

# Trucks, Cars, Garbage truck, Trailer Truck, Pickup Truck
class_labels = ['/m/07r04', '/m/0k4j', '/m/02cv4w', '/m/078my', '/m/0cvq3'] 
exclude_class = ['/m/07jdr'] # Trains
num_samples = 1000

image_ids = set()
with open(csv_file, newline='') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        image_id, label = row[0], row[2]
        if label in class_labels and label not in exclude_class:
            image_ids.add(image_id)

# Convert image_ids to a list and randomly select num_samples image ids
image_ids = random.sample(list(image_ids), num_samples)

output_file = 'ImageListTrainingNegative.txt'
if os.path.exists(output_file):
    overwrite = input(f'{output_file} already exists. Do you want to overwrite it? (y/n): ')
    if overwrite.lower() != 'y':
        print('Exiting without overwriting the file.')
        exit()

with open(output_file, 'w') as f:
    for image_id in image_ids:
        f.write(f'train/{image_id}\n')
