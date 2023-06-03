# Program to monitor a live feed of a train camera and alert when a train is on the
# tracks. Ultimately want to send a message, text, email, or something between certain times.
# This currently creates a folder for screenshots and saves them based on its prediction about whether a  
# train is present in the image. Set to take a screenshot every 1 minute. 

import os
import smtplib
import time
import torch
import io
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
screenshot_frequency = 60

# Chrome options required to prevent timeout, not sure why though.
options = Options()
options.add_argument('--headless')
options.add_argument('--window-size=1920x1080')
options.add_argument('--disable-gpu')
chrome_driver_path = os.getcwd() + "/chromedriver"
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)
driver.set_page_load_timeout(120) # Increase if timeout continues to occur.
driver.implicitly_wait(10)

# URL for the live train camera feed. This is Jefferson Parish, LA - Central Ave. 
url = "https://g1.ipcamlive.com/player/player.php?alias=63609c3400e64&autoplay=1" 

# Set size of screenshot display menu:        
window_x = 1500
window_y = 700

save_screenshots = 1 # Set to 1 to save screenshots.
show_screenshots = 1 # Set to 1 to show screenshots.
include_url = 0 # Set to 1 to include the url in the text.
include_prediction = 0 # Set to 1 to include probabilities and predictions on screenshots.

# Set the screenshot save folder:
positive_screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainingScreenshots")
if not os.path.exists(positive_screenshot_dir):
    os.makedirs(positive_screenshot_dir)
negative_screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainingScreenshotsNegative")
if not os.path.exists(negative_screenshot_dir):
    os.makedirs(negative_screenshot_dir)

root = tk.Tk()
root.geometry(f'+{window_x}+{window_y}')
label = tk.Label(root)
label.pack()
train_start_time = None
train_logged = False
train_detected = False

model = MyModel(num_classes=2)
model.load_state_dict(torch.load('TrainedModel.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
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
        if presence_prediction.item() == 0:
            print(f'Train Present - Probability of Prediction: {probability * 100:.2f}%')
        else:
            print(f'Train Not Present - Probability of Prediction: {probability * 100:.2f}%')
    if presence_prediction.item() == 0:
        print("Match! Train being logged...")
        date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        length = 0
        return True, probability
    
    print("Not a match. Moving on...")
    return False, probability

def log_train_data(date, current_time, length):
    print("Logging Train Data...")
    filename = "TrainData.csv"
    header = ["Date", "Time", "Length of Time Logged (min.)"]
    file_exists = os.path.isfile(filename)
    with open(filename, "a") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([date, current_time, length])

def update_image(image):
    photo = ImageTk.PhotoImage(image)
    label.config(image=photo)
    label.image = photo

def send_text_message(subject, body, train_duration=None, url=None):
    email_address = os.environ['EMAIL_ADDRESS']
    email_password = os.environ['EMAIL_PASSWORD']
    recipient_phone_number = os.environ['RECIPIENT_PHONE_NUMBER']
    carrier_gateway_address = 'txt.att.net'
    recipient_address = f'{recipient_phone_number}@{carrier_gateway_address}'
    
    if train_duration:
        body += f'\n\nTrain has been present for {current_train_duration:.2f} minutes.'
    if include_url == 1:
        body += f'\n\n{url}'
    
    message = f'Subject: {subject}\n\n{body}'
    
    with smtplib.SMTP('smtp.office365.com', 587) as server:
        server.starttls()
        server.login(email_address, email_password)
        server.sendmail(email_address, recipient_address, message)
    
    print('Sent text message')

while True:
    driver.get(url)
    time.sleep(10)
    screenshot = driver.get_screenshot_as_png()
    image = Image.open(io.BytesIO(screenshot))
    open_cv_image = np.array(image.convert('RGB'))
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    current_time = datetime.now()
    current_train_duration = None

    if train_start_time:
        current_train_duration = ((current_time - train_start_time) / 60).total_seconds()
    timestamp = current_time.strftime("%Y%m%d-%H%M%S")
    train_present, probability = analyze_screenshot(image, timestamp)
    
    if train_present:
        if save_screenshots == 1:
            cv2.imwrite(f'{positive_screenshot_dir}/{timestamp}.jpg', open_cv_image)
        if not train_detected:
            if 'EMAIL_ADDRESS' in os.environ and 'EMAIL_PASSWORD' in os.environ and 'RECIPIENT_PHONE_NUMBER' in os.environ:
                send_text_message('Train Alert', 'Train Alert', current_train_duration, url)
            train_detected = True
        if train_start_time is None:
            train_start_time = current_time
            if not train_logged:
                date = train_start_time.strftime("%Y-%m-%d")
                current_time_str = train_start_time.strftime("%H:%M:%S")
                log_train_data(date, current_time_str, 'NA()')
                train_logged = True
    else:
        if save_screenshots == 1:
            cv2.imwrite(f'{negative_screenshot_dir}/{timestamp}.jpg', open_cv_image)
        if train_start_time is not None:
            train_end_time = current_time
            train_duration = ((train_end_time - train_start_time) / 60).total_seconds()
            date = train_start_time.strftime("%Y-%m-%d")
            current_time_str = train_end_time.strftime("%H:%M:%S")
            log_train_data(date, current_time_str, train_duration)
            train_start_time = None
            train_logged = False
        train_detected = False

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    text = f'Probability: {probability * 100:.2f}%'
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
    text_offset_x = 10
    text_offset_y = open_cv_image.shape[0] - 10
    
    if include_prediction == 1:
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 10, text_offset_y - text_height - 10))
        cv2.rectangle(open_cv_image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
        cv2.putText(open_cv_image, text, (text_offset_x + 5, text_offset_y - 5), font, font_scale, (0, 0, 0), 2)

    # Comment out these 8 lines to stop showing screenshots.
    if save_screenshots == 1:
        new_width = 600
        original_width, original_height = image.size
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
        image = image.resize((new_width, new_height))
        update_image(image)
        root.update_idletasks()
        root.update()

time.sleep(screenshot_frequency)

