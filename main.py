import cv2
import os

# Set proxy at runtime
os.environ['HTTP_PROXY'] = 'http://edcguest:edcguest@172.31.100.25:3128'
os.environ['HTTPS_PROXY'] = 'http://edcguest:edcguest@172.31.100.25:3128'

thres = 0.45  # Threshold to detect object

# Load class names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model
configPath = 'ssd_mobilenet.pbtxt'
weightsPath = 'frozen_inference.pb'


net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Start webcam
cap = cv2.VideoCapture(0)  # Use 1 if you have an external cam
cap.set(3, 1280)     #width
cap.set(4, 720)      #height
cap.set(10, 70)      #brightness
