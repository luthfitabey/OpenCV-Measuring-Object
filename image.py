## Kelompok PKL SV-UGM ##

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils.video import FPS
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import time
import cv2
import math


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

########function #####################
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
print("[INFO] Loading model...")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor", "lorry"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe(r"C:\Users\hp\Documents\Harbour_Scanner\MobileNetSSD_deploy.prototxt.txt", r"C:\Users\hp\Documents\Harbour_Scanner\MobileNetSSD_deploy.caffemodel")

#cari titik tengah
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
#################################

################## Buka Kamera ##############
# print("[INFO] starting video stream...")

#Buka webcam
# vs = VideoStream(src=0).start()

#Buka Ipcam
# vs = VideoStream(src="http://192.168.200.105:4747/mjpegfeed?640x480").start()

#Buka USB
# vs = VideoStream(src="http://0.0.0.0:4747/mjpegfeed?640x480").start()
# 
# time.sleep(2.0)
# fps = FPS().start()
#########################################


############# Cari blob ######################
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	image = cv2.imread(args["image"])
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > 0.2 :
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			# cv2.rectangle(frame, (startX, startY), (endX, endY),
			# 	COLORS[idx], 2)
			# y = startY - 15 if startY - 15 > 15 else startY + 15
			# cv2.putText(frame, label, (startX, y),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)



################## Mencari size #################################

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-w", "--width", type=float, required=True,
# 	help="width of the left-most object in the image (in inches)")
# args = vars(ap.parse_args())

	label = "{} | {:.0f}%".format(CLASSES[idx],
			confidence * 100)

	# load the image, convert it to grayscale, and blur it slightly
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)
	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	edged = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)
	# find contours in the edge map
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	# sort the contours from left-to-right and initialize the
	# 'pixels per metric' calibration variable
	(cnts, _) = contours.sort_contours(cnts)
	pixelsPerMetric = None

	# loop over the contours individually
	for c in cnts:
		# if the contour is not sufficiently large, ignore it
		if cv2.contourArea(c) < 100:
			continue
 
		# compute the rotated bounding box of the contour		
		orig = frame.copy()
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")

		# order the points in the contour such that they appear
		# in top-left, top-right, bottom-right, and bottom-left
		# order, then draw the outline of the rotated bounding
		# box
		box = perspective.order_points(box)
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	 
		# loop over the original points and draw them
		for (x, y) in box:
			cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

		# unpack the ordered bounding box, then compute the midpoint
		# between the top-left and top-right coordinates, followed by
		# the midpoint between bottom-left and bottom-right coordinates
		(tl, tr, br, bl) = box
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)
	 
		# hitung midpoint antara titik kiri atas dan kanan atas,
		# lalu midpoint titik kanan atas dan kanan bawah
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
	 
		# draw the midpoints on the image
		cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	 
		# draw lines between the midpoints
		cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
			(255, 0, 255), 2)
		cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
			(255, 0, 255), 2)

		# compute the Euclidean distance between the midpoints
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
	 
		# if the pixels per metric has not been initialized, then
		# compute it as the ratio of pixels to supplied metric
		# (in this case, inches)
		if pixelsPerMetric is None:
			pixelsPerMetric = dB / (math.tan(90) * 7.87)
		# if pixelsPerMetric is None:
		# 	pixelsPerMetric = dB / (math.tan(["corner"]) * ["distance"])

		# if pixelsPerMetric is None:
			# pixelsPerMetric = dB / {"width"}

#### Hitung ukuran ke cm/meter/inch####

		#default inch
		dimA = (dA / pixelsPerMetric) 
		dimB = (dB / pixelsPerMetric) 
		dimC = (dimA * dimB)

		#cm
		# dimA = (dA / pixelsPerMetric) * 2.54
		# dimB = (dB / pixelsPerMetric) * 2.54
		# dimC = (dimA * dimB)

		#meter
		# dimA = (dA / pixelsPerMetric) * 0.0254
		# dimB = (dB / pixelsPerMetric) * 0.0254
		# dimC = (dimA * dimB)
	 
		# draw the object sizes on the image
		cv2.putText(orig, "{:.1f}cm".format(dimA),
			(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (255, 255, 255), 2)
		cv2.putText(orig, "{:.1f}cm".format(dimB),
			(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (255, 255, 255), 2)

#ROI Line
	# cv2.line(orig, (30, 0), (30,400), (0, 0, 0xFF), 2)	1
# output text 
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.rectangle(orig, (250, 250), (400, 450), (180, 132, 109), -1)
	cv2.putText(orig, '-Luas: ' + "{:.2f}".format(dimC), (255, 260), font, 0.4, (0xFF, 0xFF, 0x00), 1, cv2.FONT_HERSHEY_SIMPLEX)
	cv2.putText(orig, '-Harga: ' + "Rp. {:.0f}".format(dimC*10000), (255, 275), font, 0.4, (0xFF, 0xFF, 0x00), 1, cv2.FONT_HERSHEY_SIMPLEX)
	cv2.putText(orig, '-Tipe: ' + format(label), (255, 290), font, 0.4, (0xFF, 0xFF, 0x00), 1, cv2.FONT_HERSHEY_SIMPLEX)
	
# show the output frame
	cv2.imshow("Frame", orig)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("x"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()