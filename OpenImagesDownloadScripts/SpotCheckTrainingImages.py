# Diplays random images from a dataset and illustrates the bounding boxes and image classes
# of those images. Requires an annotation csv file named ClassAnnotations.csv and a folder
# of images named TrainImages in the same directory as the script.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.annotations = pd.read_csv(annotation_file)
        self.label_map = {label: index for index, label in enumerate(self.annotations['LabelName'].unique())}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_id = self.annotations.iloc[idx, 0]
        label_name = self.annotations.iloc[idx, 2]
        class_index = self.label_map[label_name]
        image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, class_index
        except FileNotFoundError:
            print(f"File not found: {image_path}")
            return torch.zeros(3, 224, 224), 0

    def get_num_classes(self):
        return len(self.label_map)

def display_random_images(dataset, num_images=5):
    num_data = len(dataset)
    indices = random.sample(range(num_data), num_images)
    fig, axs = plt.subplots(1, num_images, figsize=(15, 3))
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        image_id = dataset.annotations.iloc[idx, 0]
        label_name = dataset.annotations.iloc[idx, 2]
        class_index = dataset.label_map[label_name]
        image_path = os.path.join(dataset.image_dir, f'{image_id}.jpg')
        width, height = Image.open(image_path).size
        image = image.numpy().transpose((1, 2, 0))
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = (image * std) + mean
        
        new_size = (224, int(height * 224 / width)) if width > height else (int(width * 224 / height), 224)
        resized_image = Image.fromarray((image * 255).astype('uint8')).resize(new_size)
        axs[i].imshow(resized_image)
        
        axs[i].set_title(f'Label: {label_name}')
        axs[i].axis('off')
        annotations = dataset.annotations[dataset.annotations.iloc[:, 0] == image_id]
        for _, row in annotations.iterrows():
            xmin = row['XMin'] * width
            xmax = row['XMax'] * width
            ymin = row['YMin'] * height
            ymax = row['YMax'] * height
            new_width, new_height = resized_image.size
            xmin = int(xmin * new_width / width)
            xmax = int(xmax * new_width / width)
            ymin = int(ymin * new_height / height)
            ymax = int(ymax * new_height / height)

            rect = patches.Rectangle((xmin,ymin),xmax-xmin,
                                     ymax-ymin,
                                     linewidth=1,
                                     edgecolor='lime',
                                     facecolor='none')

            axs[i].add_patch(rect)
            axs[i].text(xmin,
                        ymin,
                        row[2],
                        fontsize=8,
                        color='white',
                        bbox=dict(facecolor='red', alpha=0.5))
    plt.show()


data_transforms = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = MyDataset(
	image_dir='TrainingImagesNegative',
	annotation_file='TrainingAnnotationsNegative.csv',
	transform=data_transforms)
display_random_images(dataset)
