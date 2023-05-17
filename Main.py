import numpy as np
import pygame
import cv2
import pickle 
from PIL import Image
import train

def cam():
    pygame.init()
    camera = cv2.VideoCapture(0)
    loaded_model = pickle.load(open('trained_model.pkl', 'rb'))   
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    pretrained = train.PretrainModel()
    while True:
        _, frame = camera.read()
        pil_image = Image.fromarray(frame)
        X = np.array(list(pretrained.get_feature(pil_image))).reshape(1,-1)
        i = int(loaded_model.predict(X))
        if i ==1:
            color_recognition(frame)
            current_time = pygame.time.get_ticks()
            while current_time >= 5000:
                current_time -=5000
            if current_time >=4900:
                alert()
        
        cv2.rectangle(frame, (100, 100), (500,400), (25, 25, 25), 3)
        cv2.resize(frame, (800, 600))
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def alert():
    pygame.mixer.init()
    pygame.mixer.music.load('warning.mp3')
    pygame.mixer.music.play()

def color_recognition(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width, _ = frame.shape

    cx = int(width / 2)
    cy = int(height / 2)

    pixel_center = hsv_frame[cy, cx]
    hue, saturation, value  = pixel_center

    color = "Unknown"
    if saturation<15 and value>215:
        color = "WHITE"
    elif hue < 5:
        color = "RED"
    elif hue < 22:
        color = "ORANGE"
    elif hue < 33:
        color = "YELLOW"
    elif hue < 78:
        color = "GREEN"
    elif hue  < 131:
        color = "BLUE"
    elif hue < 170:
        color = "VIOLET"
    else:
        color = "RED"

    pixel_center_bgr = frame[cy, cx]
    b, g, r = int(pixel_center_bgr[0]), int(pixel_center_bgr[1]), int(pixel_center_bgr[2])
    cv2.putText(frame, color, (cx - 200, 80), 0, 3, (b, g, r), 5)

cam()