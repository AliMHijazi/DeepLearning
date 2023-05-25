# Program to monitor a live feed of a train camera and alert when a train is on the
# tracks. Ultimately want to send a message, text, email, or something between certain times.
# This currently creates a folder for screenshots and saves them. 
# Set to take a screenshot every 5 minutes. 

import time
import torch
import io
import os
import cv2
import csv
import numpy as np
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from PIL import Image
from torchvision import transforms
from DataSetTrainer import MyModel


print("Initializing...")
options = Options()
options.add_argument("--headless")
options.add_argument("--window-size=1920x1080")
chrome_driver_path = os.getcwd() + "/chromedriver"
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)

# URL for the live train camera feed. This is Jefferson Parish, LA - Central Ave. 
url = "https://g1.ipcamlive.com/player/player.php?alias=63609c3400e64&autoplay=1" 

screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Screenshots")
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)

model = MyModel(num_classes=1)
model.load_state_dict(torch.load('TrainedModel.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225] # Adjust this?
    )
])

def analyze_screenshot(screenshot):
    print("Analyzing Screenshot...")
    screenshot = screenshot.convert('RGB')
    screenshot = transform(screenshot)
    screenshot = screenshot.unsqueeze(0)
    with torch.no_grad():
        output = model(screenshot)
        presence_prediction = torch.argmax(output, dim=1)
    if presence_prediction.item() == 1: # Adjust this?
        print("Match! Train being logged...")
        date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Need to add something here to log the length of time of the train
        length = 10 
        
        # Need to add something here to log the color of the train

        log_train_data(date, current_time, length) # Add color too
        return True
    
    print("Not a match. Moving on...")
    return False

def log_train_data(date, current_time, length):
    print("Logging Train Data...")
    with open("TrainData.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([date, current_time, color, length])

while True:
    driver.get(url)
    time.sleep(5)
    screenshot = driver.get_screenshot_as_png()
    image = Image.open(io.BytesIO(screenshot))
    
    # Comment out two lines below to not save screenshots
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    image.save(f'{screenshot_dir}/{timestamp}.png')

    train_present = analyze_screenshot(image)
    time.sleep(60)
