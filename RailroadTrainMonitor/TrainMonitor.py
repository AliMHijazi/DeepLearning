# Program to monitor a live feed of a train camera and alert when a train is on the tracks.
# Requires a trained model named TrainedModel.pth in the folder with the program. 
# Has on/off functionality for sending texts, saving screenshots, showing screenshots, 
# including the url in the text, and including the prediction on the screenshots.
# Text functionality requires that the config.ini file be set with the user's email and phone info. 
# Gateway address and SMTP info should be changed accordinly as well. 

import os
import smtplib
import time
import torch
import io
import cv2
import csv
import signal
import sys
import configparser
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

# Chrome options required to prevent timeout.
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

window_x = 1500 # Set size of screenshot display menu.    
window_y = 700
screenshot_frequency = 60 # Seconds
send_texts = 1 # Set to 1 and save env. variables to send texts.
save_screenshots = 0 # Set to 1 to save screenshots.
show_screenshots = 0 # Set to 1 to show screenshots.
include_url = 1 # Set to 1 to include the url in the text message.
include_prediction = 0 # Set to 1 to include probabilities and predictions on screenshots.
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'TrainedModel.pth')

config = configparser.ConfigParser()
config.read('config.ini')
email_address = config['DEFAULT']['Email_Address']
email_password = config['DEFAULT']['Email_Password']
recipient_phone_number = config['DEFAULT']['Recipient_Phone_Number']

print(f"Email address: {email_address}")
print(f"Recipient phone number: {recipient_phone_number}")

# Set the screenshot save folder:
positive_screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainingScreenshots")
if not os.path.exists(positive_screenshot_dir):
    os.makedirs(positive_screenshot_dir)
negative_screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainingScreenshotsNegative")
if not os.path.exists(negative_screenshot_dir):
    os.makedirs(negative_screenshot_dir)
if show_screenshots == 1:
    root = tk.Tk()
    root.geometry(f'+{window_x}+{window_y}')
    label = tk.Label(root)
    label.pack()
else:
    class DummyTk:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    root = DummyTk()
    
    class DummyLabel:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    label = DummyLabel()

shutdown_flag = False
model = MyModel(num_classes=2)
model.load_state_dict(torch.load(model_path))
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

def handle_sigterm(signum, frame):
    driver.quit()
    sys.exit(0)
    
signal.signal(signal.SIGTERM, handle_sigterm)

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
        date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        length = 0
        return True, probability
    return False, probability

def log_train_data(date, current_time, length):
    filename = "TrainData.csv"
    header = ["Date", "Time", "Length of Time Logged (min.)"]
    file_exists = os.path.isfile(filename)
    with open(filename, "a") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([date, current_time, length])

def update_image(image):
    if show_screenshots == 1:
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo
    else:
        photo = None
        
def send_text_message(subject, body, email_address, email_password, recipient_phone_number, train_duration=None, url=None):
    carrier_gateway_address = 'txt.att.net' # Change accordingly.
    recipient_address = f'{recipient_phone_number}@{carrier_gateway_address}'
    # Still plan to use train duration for something. Just not sure where.     
    if train_duration:
        body += f'\n\nTrain has been present for {current_train_duration:.2f} minutes.'
    if include_url == 1:
        body += f'\n\n{url}'
    message = f'Subject: {subject}\n\n{body}'
    with smtplib.SMTP('smtp.office365.com', 587) as server: # Change accordingly.
        server.starttls()
        server.login(email_address, email_password)
        server.sendmail(email_address, recipient_address, message)
    print('Sent text message')
    
def shutdown():
    global shutdown_flag
    shutdown_flag = True
    print('Shutting Down...')
    driver.close()
    root.destroy()

if show_screenshots == 1:
    shutdown_button = tk.Button(root, text="Shutdown", command=shutdown)
    shutdown_button.pack()

def on_key_press(event):
    print(f'Key pressed: {event.char}')
    if event.char == 'q':
        global shutdown_flag
        shutdown_flag = True
        print('Q pressed')
        driver.close()
        root.destroy()

start_time = time.time()
driver.get(url)
root.bind('<Key>', on_key_press)
train_detected = False
train_start_time = None
def update():
    global train_detected
    global train_start_time
    text = ""
    screenshot_buffer = 15 # Adjust up if screenshots are still loading.
    load_start_time = datetime.now()    
    train_logged = False
    driver.refresh()
    time.sleep(screenshot_buffer)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    text_thickness = 2
    screenshot = driver.get_screenshot_as_png()
    image = Image.open(io.BytesIO(screenshot))
    open_cv_image = np.array(image.convert('RGB'))
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    current_time = datetime.now()
    current_train_duration = None
    if train_start_time:
        current_train_duration = ((current_time - train_start_time) / 60).total_seconds()
    timestamp = current_time.strftime("%Y%m%d-%H%M%S")
    root.title(f'Screenshot: {timestamp}')
    train_present, probability = analyze_screenshot(image, timestamp)
    if include_prediction == 1:
        if train_present:
            text = f'Train Detected - Probability: {probability * 100:.2f}%'
        else:
            text = f'No Train - Probability: {probability * 100:.2f}%'
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=text_thickness)[0]
        text_offset_x = 10
        text_offset_y = open_cv_image.shape[0] - 10
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 10, text_offset_y - text_height - 10))
        cv2.rectangle(open_cv_image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
        cv2.putText(open_cv_image, text, (text_offset_x + 5, text_offset_y - 5), font, font_scale, (0, 0, 0), 2)
    if train_present:
        if save_screenshots == 1:
            cv2.imwrite(f'{positive_screenshot_dir}/{timestamp}.jpg', open_cv_image)
            new_width = 600
            original_width, original_height = image.size
            aspect_ratio = original_height / original_width
            new_height = int(new_width * aspect_ratio)
            image = image.resize((new_width, new_height))
            if show_screenshots == 1:
                update_image(image)
            root.update_idletasks()
            root.update()
        if not train_detected:
            if send_texts == 1 and email_address and email_password and recipient_phone_number:
                send_text_message('Train Alert', 'Train Alert', email_address, email_password, recipient_phone_number, current_train_duration, url)
            train_detected = True
        if train_start_time is None:
            train_start_time = current_time
            if not train_logged:
                date = train_start_time.strftime("%Y-%m-%d")
                current_time_str = train_start_time.strftime("%H:%M:%S")
                print("Logging time train identified...")
                log_train_data(date, current_time_str, 'NA()')
                train_logged = True
    else:
        if save_screenshots == 1:
            new_width = 600
            original_width, original_height = image.size
            aspect_ratio = original_height / original_width
            new_height = int(new_width * aspect_ratio)
            image = image.resize((new_width, new_height))
            update_image(image)
            root.update_idletasks()
            root.update()
            cv2.imwrite(f'{negative_screenshot_dir}/{timestamp}.jpg', open_cv_image)
        if train_start_time is not None:
            train_end_time = current_time
            train_duration = ((train_end_time - train_start_time) / 60).total_seconds()
            date = train_start_time.strftime("%Y-%m-%d")
            current_time_str = train_end_time.strftime("%H:%M:%S")
            print("Logging train end time...")
            log_train_data(date, current_time_str, train_duration)
            train_start_time = None
            train_logged = False
        train_detected = False
    load_time = datetime.now() - load_start_time
    if not shutdown_flag:
        root.after((screenshot_frequency - int(load_time.total_seconds())) * 1000, update)
update()
no_screen_load_time = time.time() - start_time
if show_screenshots == 1:
    root.mainloop()
else:
    while True:
        start_time = time.time()
        update()
        load_time = time.time() - start_time
        time.sleep(max(0, screenshot_frequency - load_time))

