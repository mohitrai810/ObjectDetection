import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Load the pre-trained YOLOv3 model
net = cv2.dnn.readNetFromDarknet("C:\\Users\\mohit\\OneDrive\\Desktop\\ObjectDetection\\raw.githubusercontent.com_pjreddie_darknet_master_cfg_yolov3.cfg.txt", "C:\\Users\\mohit\\OneDrive\\Desktop\\ObjectDetection\\yolov3.weights")

# Load the class labels
classes = []
with open("C:\\Users\\mohit\\OneDrive\\Desktop\\ObjectDetection\\raw.githubusercontent.com_pjreddie_darknet_master_data_coco.names.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Create a Tkinter window to browse and select the image file
root = tk.Tk()
root.withdraw()

# Ask the user to select an image file
image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

# Check if an image is selected
if not image_path:
    print("No image selected.")
    exit()

# Load the image
image = cv2.imread(image_path)

# Check if the image is successfully loaded
if image is None:
    print("Failed to load the image.")
    exit()

# Obtain the dimensions of the image
height, width, _ = image.shape

# Create a blob from the image and perform forward pass
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
output_layers = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers)

# Define the confidence threshold and non-maximum suppression threshold
confidence_threshold = 0.5
nms_threshold = 0.4

# Process the detections
boxes = []
confidences = []
class_ids = []

for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > confidence_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression to remove overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

# Check if there are detections
if len(indices) > 0:
    # Draw bounding boxes and labels
    for i in indices.flatten():
        box = boxes[i]
        x, y, w, h = box
        label = classes[class_ids[i]]
        confidence = confidences[i]

        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Put label and confidence on the box
        text = f'{label}: {confidence:.2f}'
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image
cv2.imshow('Multi-Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
