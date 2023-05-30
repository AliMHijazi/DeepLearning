# Program to monitor a live feed of a train camera and alert when a train is on the
# tracks. Ultimately want to send a message, text, email, or something between certain times.
# This currently creates a folder for screenshots and saves them based on its prediction about whether a  
# train is present in the image. Set to take a screenshot every 1 minute. 

import time
import torch
import io
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from PIL import Image
from PIL import ImageTk
from torchvision import transforms
from DatasetTrainer import MyModel


print("Initializing...")
# One of these is required to prevent timeout, not sure which one though.
options = Options()
options.add_argument('--headless')
options.add_argument('--window-size=1920x1080')
options.add_argument('--disable-gpu')
"""options.add_argument("--disable-browser-side-navigation")
options.add_argument("--no-sandbox")
options.add_argument("--disable-extensions")
options.add_argument("--dns-prefetch-disable")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--aggressive-cache-discard")
options.add_argument("--disable-cache")
options.add_argument("--disable-application-cache")
options.add_argument("--disable-offline-load-stale-cache")
options.add_argument("--disk-cache-size=0")
options.add_argument("--no-proxy-server")
options.add_argument("--ignore-certificate-errors")
options.add_argument("--disable-notifications")
options.add_argument("--mute-audio")
options.add_argument("--enable-automation")
options.add_argument("--disable-features=NetworkService")"""
chrome_driver_path = os.getcwd() + "/chromedriver"
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)
driver.set_page_load_timeout(120) # Increase if timeout continues to occur.
driver.implicitly_wait(10)

# URL for the live train camera feed. This is Jefferson Parish, LA - Central Ave. 
url = "https://g1.ipcamlive.com/player/player.php?alias=63609c3400e64&autoplay=1" 
screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Screenshots")
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)
positive_screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainingScreenshots")
if not os.path.exists(positive_screenshot_dir):
    os.makedirs(positive_screenshot_dir)
negative_screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainingScreenshotsNegative")
if not os.path.exists(negative_screenshot_dir):
    os.makedirs(negative_screenshot_dir)

model = MyModel(num_classes=2)
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

def analyze_screenshot(screenshot, filename):
    print(f'Analyzing Screenshot: {filename}.jpg')
    screenshot = screenshot.convert('RGB')
    screenshot = transform(screenshot)
    screenshot = screenshot.unsqueeze(0)
    with torch.no_grad():
        output = model(screenshot)
        probabilities = torch.softmax(output, dim=1)
        presence_prediction = torch.argmax(probabilities, dim=1)
        probability = probabilities[0][presence_prediction].item()
        print(f'Raw model output: {output}')
        print(f'Probabilities: Class 0: {probabilities[0][0] * 100:.2f}%, Class 1: {probabilities[0][1] * 100:.2f}%')
        print(f'Predicted class: {presence_prediction.item()}')
        print(f'Probability: {probability * 100:.2f}%')
    if presence_prediction.item() == 0:
        print("Match! Train being logged...")
        date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Need to add something here to log the length of time of the train
        length = 10 
        
        # Need to add something here to log the color of the train

        log_train_data(date, current_time, length) # Add color too
        return True, probability
    
    print("Not a match. Moving on...")
    return False, probability

def log_train_data(date, current_time, length):
    print("Logging Train Data...")
    with open("TrainData.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([date, current_time, length])
        
window_x = 150
window_y = 100
root = tk.Tk()
root.geometry(f'+{window_x}+{window_y}')
label = tk.Label(root)
label.pack()

def update_image(image):
    photo = ImageTk.PhotoImage(image)
    label.config(image=photo)
    label.image = photo

while True:
    driver.get(url)
    time.sleep(10)
    screenshot = driver.get_screenshot_as_png()
    image = Image.open(io.BytesIO(screenshot))
    open_cv_image = np.array(image.convert('RGB'))
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_present, probability = analyze_screenshot(image, timestamp)
    
    if train_present:
        cv2.imwrite(f'{positive_screenshot_dir}/{timestamp}.jpg', open_cv_image)
    else:
        cv2.imwrite(f'{negative_screenshot_dir}/{timestamp}.jpg', open_cv_image)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    text = f'Probability: {probability * 100:.2f}%'
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
    text_offset_x = 10
    text_offset_y = open_cv_image.shape[0] - 10
    
#   box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 10, text_offset_y - text_height - 10))
#   cv2.rectangle(open_cv_image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
#   cv2.putText(open_cv_image, text, (text_offset_x + 5, text_offset_y - 5), font, font_scale, (0, 0, 0), 2)

    new_width = 600
    original_width, original_height = image.size
    aspect_ratio = original_height / original_width
    new_height = int(new_width * aspect_ratio)
    image = image.resize((new_width, new_height))
    update_image(image)
    #label.place(x=500, y=100)
    root.update_idletasks()
    root.update()
    time.sleep(60)
