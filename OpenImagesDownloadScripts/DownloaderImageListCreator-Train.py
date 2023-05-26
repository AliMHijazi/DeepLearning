# This script will take the annotations .csv file and create a text file 
# called ImageList.txt for use with the Downloader.py script from
# OpenImages which is available at this URL:
# https://raw.githubusercontent.com/openimages/dataset/master/downloader.py
# This will create the downloader file for the TRAINING set. 

import csv
csv_file = 'oidv6-train-annotations-bbox.csv'

# Change class_label based on the class label of interest
# https://storage.googleapis.com/openimages/2017_07/class-descriptions.csv
class_label = '/m/07jdr' # Railroad Trains
image_ids = set()
with open(csv_file, newline='') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        image_id, label = row[0], row[2]
        if label == class_label:
            image_ids.add(image_id)
with open('ImageListTrain.txt', 'w') as f:
    for image_id in image_ids:
        f.write(f'train/{image_id}\n')
