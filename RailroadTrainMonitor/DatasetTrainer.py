# Makes a trained model based on a folder of images and the associated annotations file. 
# Needs a bit of tweaking for sure. 

import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MyDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, negative_annotation_file=None):
        self.image_dir = image_dir
        self.transform = transform
        self.annotations = pd.read_csv(annotation_file)
        if negative_annotation_file:
            negative_annotations = pd.read_csv(negative_annotation_file)
            # Set the label for the negative examples
            negative_annotations['LabelName'] = 'Negative'
            # Concatenate the positive and negative annotations
            self.annotations = pd.concat([self.annotations, negative_annotations], ignore_index=True)
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

class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Need to adjust these?
if __name__ == "__main__":

    def get_accuracy(model, data_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total


    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MyDataset(
        image_dir='TrainingImages',
        annotation_file='TrainingAnnotations.csv',
        transform=data_transforms,
        negative_annotation_file='TrainingAnnotationsNegative.csv')

    num_classes = train_dataset.get_num_classes()
    model = MyModel(num_classes=num_classes)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    val_dataset = MyDataset(
        image_dir='TestingImages',
        annotation_file='TestingAnnotations.csv',
        transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # Need to adjust these?
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                train_acc = get_accuracy(model, train_loader)
                val_acc = get_accuracy(model, val_loader)
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f} train acc: {train_acc:.2f}% val acc: {val_acc:.2f}%')
                running_loss = 0.0

    torch.save(model.state_dict(), 'TrainedModel2.pth')
    print('Finished Training!')
