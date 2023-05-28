import os
import csv
training_images_path = 'TrainingImages'
training_images_negative_path = 'TrainingImagesNegative'
image_files = [f for f in os.listdir(training_images_path) if f.endswith('.jpg') or f.endswith('.png')]
with open('TrainingAnnotations.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'])
    for image_file in image_files:
        image_id = os.path.splitext(image_file)[0]
        writer.writerow([image_id, 'xclick', '/m/07jdr', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
image_files_negative = [f for f in os.listdir(training_images_negative_path) if f.endswith('.jpg') or f.endswith('.png')]
with open('TrainingAnnotationsNegative.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'])
    for image_file in image_files_negative:
        image_id = os.path.splitext(image_file)[0]
        writer.writerow([image_id, '', '', '', '', '', '', '', '', '', '', '', ''])
