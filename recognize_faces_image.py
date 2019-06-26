# Importing the packages
import face_recognition
import argparse
import pickle
import cv2

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image" , required = True, help = "path to input image")
ap.add_argument("-e", "--encodings", required = True, help = "path to encoded image")
ap.add_argument("-d", "--detection_method", type = str, default = "cnn", help = "face detection model : 'hog' or 'cnn'")
args = vars(ap.parse_args())

# Loading the known faces and embeddings
print("[INFO] Loading encodings .... ")
data = pickle.loads(open(args["encodings"],"rb").read())

# Loading the input image and converting it to RGB from BGR
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Processing the input image
# We are calculatring the (x,y) co-ordinates of the bounding boxes corresponding to each face in the input image
# Then we will compute the facial embeddings for each face
print("[INFO] recognizing faces....")
boxes = face_recognition.face_locations(rgb,model = args["detection_method"])
encodings = face_recognition.face_encodings(rgb,boxes)

names = []

for encoding in encodings:
	# compare_faces will internally calculate the euclidean distance between the two images.
	# If the resultant value is smaller than  a threshold then True is returned otherwise False
	matches = face_recognition.compare_faces(data["encodings"],encoding)
	name = "Unknown"

	if True in matches:

		# Finding all the indices which have True value and storing them in matchedIdxs
		matchedIdxs = [i for (i,b) in enumerate(matches) if b]
		counts = {}

		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name,0) + 1

		name = max(counts,key = counts.get)

	names.append(name)

for ((top,right,bottom,left),name) in zip(boxes,names):
	cv2.rectangle(image, (left,top) , (right,bottom) , (0,255,0) , 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)



