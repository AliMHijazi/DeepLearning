# This will take the full .csv file of the annotations from OpenImages and will output 
# a csv file with only those images listed with the class_label below.
# The csv files can be found at this url:
# https://storage.googleapis.com/openimages/web/download_v7.html

import csv
csv_file = 'oidv6-train-annotations-bbox.csv'

# Change class_label based on the class label of interest
# https://storage.googleapis.com/openimages/2017_07/class-descriptions.csv
class_label = '/m/07jdr' # Railroad Trains

with open(csv_file, newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    with open('ClassAnnotations.csv', 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(header)
        for row in reader:
            image_id, label = row[0], row[2]
            if label == class_label:
                writer.writerow(row)
