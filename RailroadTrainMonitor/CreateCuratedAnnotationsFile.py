# This is to be used if the TrainingImages folder has been hand curated
# and the annotations file needs to be trimmed. Writes a new .csv file 
# called TrainingAnnotationsCurated.csv.

import os
import csv
images_path = 'TrainingImages'
annotations_path = 'TrainingAnnotations.csv'
new_annotations_path = 'TrainingAnnotationsCurated.csv'
image_filenames = os.listdir(images_path)
with open(annotations_path, 'r') as annotations_file:
    reader = csv.reader(annotations_file)
    header_row = next(reader)
    with open(new_annotations_path, 'w') as new_annotations_file:
        writer = csv.writer(new_annotations_file)
        writer.writerow(header_row)
        for row in reader:
            if row[0] + '.jpg' in image_filenames:
                writer.writerow(row)
