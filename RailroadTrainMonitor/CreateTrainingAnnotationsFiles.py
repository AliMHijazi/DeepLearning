import os
import csv

def confirm_overwrite(file_path):
    while True:
        response = input(f'{file_path} already exists. Do you want to overwrite it? (y/n): ')
        if response.lower() == 'y':
            return True
        elif response.lower() == 'n':
            return False

training_images_path = 'TrainingImages'
training_images_negative_path = 'TrainingImagesNegative'
image_files = [f for f in os.listdir(training_images_path) if f.endswith('.jpg') or f.endswith('.png')]
annotations_file_path = 'TrainingAnnotations.csv'
if os.path.exists(annotations_file_path):
    if confirm_overwrite(annotations_file_path):
        with open(annotations_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'])
            for image_file in image_files:
                image_id = os.path.splitext(image_file)[0]
                writer.writerow([image_id, 'xclick', '/m/07jdr', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
else:
    with open(annotations_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(['ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'])
        for image_file in image_files:
            image_id = os.path.splitext(image_file)[0]
            writer.writerow([image_id, 'xclick', '/m/07jdr', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
image_files_negative = [f for f in os.listdir(training_images_negative_path) if f.endswith('.jpg') or f.endswith('.png')]

annotations_negative_file_path = 'TrainingAnnotationsNegative.csv'

if os.path.exists(annotations_negative_file_path):
    if confirm_overwrite(annotations_negative_file_path):
        with open(annotations_negative_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'])
            for image_file in image_files_negative:
                image_id = os.path.splitext(image_file)[0]
                writer.writerow([image_id, '', 'Negative', '', '', '', '', '', '', '', '', '', ''])
