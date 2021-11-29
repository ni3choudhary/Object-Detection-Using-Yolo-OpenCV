# Import Necessary Packages
import cv2
import argparse
import os
import numpy as np
import sys

parser = argparse.ArgumentParser(description='Use this script to run Yolov3 object detection')
parser.add_argument('--image', help='Path to image file')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args() 

# Load names of classes/labels
LABELS = []
with open('labels.txt', 'r') as f:
    LABELS = f.read().splitlines()

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# print(LABELS)
# derive the paths to the Yolov3 weights and model configuration
weights = 'yolov3.weights'
cfg = 'yolov3.cfg'

# Initialize the parameters
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold
NMS_THRESHOLD = 0.4  # NMS threshold

# Give the configuration and weight files for the model and load the network.
print("[INFO] loading Yolov3 from disk...")
net = cv2.dnn.readNetFromDarknet(cfg, weights)

outputFile = "yoloV3_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yoloV3_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yoloV3_py.avi'
else:
    # Webcam input
    cap = cv2.VideoCapture(1)

writer = None

# loop over frames from the video file stream
while True:
	grabbed, img = cap.read()
	# H,W,_ = img.shape
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	# construct a blob from the image,which is input for the network.
	blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False) 
	# It has the following parameters-
		# the image to transform
		# the scale factor (1/255 to scale the pixel values to [0..1])
		# the size, here a 416x416 square image
		# the mean value (default=0)
		# the option swapBR=True (since OpenCV uses BGR)

	# calculate the network response:
	net.setInput(blob)
	output_layers_names = net.getUnconnectedOutLayersNames()
	layerOutputs = net.forward(output_layers_names) 

	# The layerOutputs object are vectors of length 85 which are mention below

	#4x the bounding box (centerX, centery, width, height)
	#1x box confidence
	#80x class confidence

	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the scores,class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > CONFIDENCE_THRESHOLD:
				# 0.5 is minimum probability
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				(H, W) = img.shape[:2]
				# centerX = int(detection[0] * W)
				# centery = int(detection[1] * H)
				# width = int(detection[2] * W)
				# height = int(detection[3] * H)
				

				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - width / 2)
				y = int(centerY - height / 2)
				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID) 


	# apply non-maxima suppression to suppress weak, overlapping bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,NMS_THRESHOLD)
	#Font color
	font = cv2.FONT_HERSHEY_PLAIN
	# Color for boxes and text
	COLORS = np.random.uniform(0, 255, size=(len(boxes), 3))

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# draw a bounding box rectangle and label on the frame
			color = COLORS[i]
			cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
					 	confidences[i])
			cv2.putText(img, text, (x, y - 5),
						font, 2, color, 2)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer to save the output video
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(outputFile, fourcc, 30,
			(img.shape[1], img.shape[0]), True)

	# write the output frame/image to disk
	# Write the frame/image with the detection boxes
	if (args.image):
		cv2.imwrite(outputFile,img.astype(np.uint8))
	else:
		writer.write(img.astype(np.uint8))

	# Display Object detection
	cv2.imshow('Object Detection',  img)
	
	if cv2.waitKey(25) & 0xFF == ord("q"):
		break

# release the file pointers
print("[INFO] cleaning up...")
if (args.image):
	writer.release()
cap.release()
cv2.destroyAllWindows()


