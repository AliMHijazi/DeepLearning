import os
import csv

# Set the paths to the TrainingImages and TrainingImagesNegative folders
training_images_path = 'TrainingImages'
training_images_negative_path = 'TrainingImagesNegative'

# Get a list of all the image files in the TrainingImages folder
image_files = [f for f in os.listdir(training_images_path) if f.endswith('.jpg') or f.endswith('.png')]

# Create a new CSV file called TrainingAnnotations.csv
with open('TrainingAnnotations.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header row
    writer.writerow(['ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'])
    # Write a row for each image file
    for image_file in image_files:
        # Get the image id by removing the file extension from the file name
        image_id = os.path.splitext(image_file)[0]
        # Write a row with the image id and other annotation data
        writer.writerow([image_id, 'xclick', '/m/07jdr', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Get a list of all the image files in the TrainingImagesNegative folder
image_files_negative = [f for f in os.listdir(training_images_negative_path) if f.endswith('.jpg') or f.endswith('.png')]

# Create a new CSV file called TrainingAnnotationsNegative.csv
with open('TrainingAnnotationsNegative.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header row
    writer.writerow(['ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'])
    # Write a row for each image file
    for image_file in image_files_negative:
        # Get the image id by removing the file extension from the file name
        image_id = os.path.splitext(image_file)[0]
        # Write a row with the image id and other annotation data
        writer.writerow([image_id, '', '', '', '', '', '', '', '', '', '', '', ''])
