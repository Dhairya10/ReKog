# Importing necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset" , required = True, help = "path to input directory")
ap.add_argument("-e", "--encodings", required = True, help = "path to encoded image")
ap.add_argument("-d", "--detection_method", type = str, default = "cnn", help = "face detection model : 'hog' or 'cnn'")
args = vars(ap.parse_args())

# Path of input image
print("[INFO] quantifying faces....")
imagePaths = list(paths.list_images(args["dataset"]))

# Initialising list of names and encodings
knownEncodings = []
knownNames = []

for(i,imagePath) in enumerate(imagePaths):
	# Extracting the name
	print("[INFO] processing image {} / {} ".format(i+1,len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# Converting from BGR (OpenCV format) to RGB (dlib format)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

	boxes = face_recognition.face_locations(rgb,model = args["detection_method"])

	# Computing the 128d vector for each face
	encodings = face_recognition.face_encodings(rgb,boxes)

	# Appending the encoding and names
	##### An image can have more than one face.
	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

# Dumping the encodings and names
print("[INFO] serializing encodings....")		
data = {"encodings" : knownEncodings, "names" : knownNames}
f = open(args["encodings"],"wb")
f.write(pickle.dumps(data))
f.close()
