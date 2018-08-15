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
import argparse
import math

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--unit", type=str, required=True,
	help="unit of measurement, 'm' for give you meter and 'cm' for give you centimeter" )
# ap.add_argument("-w", "--width", type=float, required=Tue,
# 	help="width of the left-most object in the image (in inches)")
# ap.add_argument("-r", "--radian", type=float, required=True,
# 	help="radian of alpha")
# ap.add_argument("-d", "--distance", type=float, required=True,
# 	help="distabce between camera and object")
args = vars(ap.parse_args())

# Loading model from opensource API
print("[INFO] Loading model...")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor", "lorry"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe(r"C:\Users\hp\Documents\Harbour_Scanner\MobileNetSSD_deploy.prototxt.txt", r"C:\Users\hp\Documents\Harbour_Scanner\MobileNetSSD_deploy.caffemodel")

#initialize midpoint
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#Open Cam
print("[INFO] starting video stream...")

#Buka webcam
vs = VideoStream(src=0).start()

#Buka Ipcam
# vs = VideoStream(src="http://192.168.200.105:4747/mjpegfeed?640x480").start()

#Buka USB
# vs = VideoStream(src="http://0.0.0.0:4747/mjpegfeed?640x480").start()

time.sleep(2.0)
fps = FPS().start()

############# search blob to get model ######################
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)

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

################## Measuring size #################################
	#get label from blob to print at imshow
	lbl = label
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
	 
		# calculate the midpoint between the top left and right upper points,
		# then midpoint right upper and lower right point
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
	 

		if args["unit"] == "cm" :
			#coba cm (bisa, bener untuk jarak <70cm)
			dimA = (dA  * 0.026458) 
			dimB = (dB * 0.026458) 
			dimC = (dimA * dimB)

		#compute the euclidean distance (px) to actual measurement
				# if pixelsPerMetric is None:
				# 	pixelsPerMetric = dB / (math.tan(90) * 7.87)

				# if pixelsPerMetric is None:
				# 	pixelsPerMetric = dB / (math.tan(["corner"]) * ["distance"])

				#salah
				# if pixelsPerMetric is None:
				# 	pixelsPerMetric = dB / args["width"] + args["distance"]

				# if pixelsPerMetric is None:
				# 	pixelsPerMetric = dB / 0.026458

				#coba disamakan satuannya ke cm (salah)
				# if pixelsPerMetric is None:
				# 	pixelsPerMetric = (dB * 0.0264583333)  / args["width"]

				#coba disamakan satuannya ke px pake sudut
				# if pixelsPerMetric is None:
				# 	pixelsPerMetric = dB  / ((args["width"] * math.tan(180)) * 37.795275591)

				#coba disamakan satuannya ke px
				# if pixelsPerMetric is None:
				# 	pixelsPerMetric = dB  / (args["width"] * 37.795275591)

		#### Hitung ukuran ke cm/meter/inch####
				#default inch (bisa)
				# dimA = (dA / pixelsPerMetric) 
				# dimB = (dB / pixelsPerMetric) 
				# dimC = (dimA * dimB)

				#cm --> disamakan satuannya (salah)
				# dimA = ((dA * 0.0264583333) / pixelsPerMetric) 
				# dimB = ((dB * 0.0264583333) / pixelsPerMetric) 
				# dimC = (dimA * dimB)

				#px --> disamakan satuannya (salah)
				# dimA = (dA / pixelsPerMetric) * 0.026458333
				# dimB = (dB / pixelsPerMetric) * 0.026458333
				# dimC = (dimA * dimB)

				#meter
				# dimA = (dA / pixelsPerMetric) * 0.0254
				# dimB = (dB / pixelsPerMetric) * 0.0254
				# dimC = (dimA * dimB)

				#hehe 
				# dimA = (dA / pixelsPerMetric) * 0.026458
				# dimB = (dB / pixelsPerMetric) * 0.026458
				# dimC = (dimA * dimB)

				#haha gagal
				# dimA = (dA  * 0.026458) + (math.tan(180) *30)
				# dimB = (dB  * 0.026458)
				# dimC = (dimA * dimB)

			# draw the object sizes on the image
			cv2.putText(orig, "{:.1f}cm".format(dimA),
				(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (255, 255, 255), 2)
			cv2.putText(orig, "{:.1f}cm".format(dimB),
				(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (255, 255, 255), 2)

			# output text 
			#dim c = dimension of object
			#price = (dima * dimb ) * Rp 10000
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.rectangle(orig, (1000, 1000), (700, 620), (800, 132, 109), -1)
			cv2.putText(orig, '-Luas: ' + "{:.2f} cm^2".format(dimC), (700, 650), font, 0.7, (0xFF, 0xFF, 0x00), 1, cv2.FONT_HERSHEY_SIMPLEX)
			cv2.putText(orig, '-Harga: ' + "Rp. {:.0f}".format(dimC*10000), (700, 690), font, 0.7, (0xFF, 0xFF, 0x00), 1, cv2.FONT_HERSHEY_SIMPLEX)
			cv2.putText(orig, '-Tipe: ' + format(lbl), (700, 730), font, 0.7, (0xFF, 0xFF, 0x00), 1, cv2.FONT_HERSHEY_SIMPLEX)
					
		elif args["unit"] == "m" :  
	        #coba meter (bisa)
			dimA = (dA * 0.000264583)
			dimB = (dB * 0.000264583)
			dimC = (dimA * dimB)
			cv2.putText(orig, "{:.1f}m".format(dimA),
				(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (255, 255, 255), 2)
			cv2.putText(orig, "{:.1f}m".format(dimB),
				(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (255, 255, 255), 2)
			# output text 
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.rectangle(orig, (1000, 1000), (700, 620), (800, 132, 109), -1)
			cv2.putText(orig, '-Luas: ' + "{:.2f} m^2".format(dimC), (700, 650), font, 0.7, (0xFF, 0xFF, 0x00), 1, cv2.FONT_HERSHEY_SIMPLEX)
			cv2.putText(orig, '-Harga: ' + "Rp. {:.0f}".format(dimC*10000), (700, 690), font, 0.7, (0xFF, 0xFF, 0x00), 1, cv2.FONT_HERSHEY_SIMPLEX)
			cv2.putText(orig, '-Tipe: ' + format(label), (700, 730), font, 0.7, (0xFF, 0xFF, 0x00), 1, cv2.FONT_HERSHEY_SIMPLEX)
				
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