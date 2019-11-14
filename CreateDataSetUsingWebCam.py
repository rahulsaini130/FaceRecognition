# Import the CV2, OS
import cv2
import os

# Directory path where images store
# FolderName - The name of the class or person
# ImageName - The name of the image
DATADIR = "E:/ML Projects/LiveFaceDetection/dataset"
FolderName = "Vijay"
ImageName = "Vijayq"

# Import the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 1. Join the path
# 2. Create folder if not exist in the path
# 3. Save the image to the given path 
def genrate_data(FolderName, ImageName, img_id, img):
    pathToDir = os.path.join(DATADIR, FolderName)
    if not os.path.exists(pathToDir):
        os.mkdir(pathToDir)
    cv2.imwrite("dataset/" + FolderName + "/" + ImageName + "-" + str(img_id) + ".jpg", img) 

# Start the webcam 
# 0 -  for the default camera
video_capture = cv2.VideoCapture(0)

# 1. Img_id is the number of frames to be taken 
# 2. Read the webcam with video_capture.read() funtion
# 3. Convert to grayScale
# 4. detect the features from the frame or image
        # gray_img - the img in gray scale 
        # 1.1 - is the scale factor
        # 10 - minNeighbor
 
img_id = 0
while True:
    img_id += 1
    _, img = video_capture.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = face_cascade.detectMultiScale(gray_img, 1.1, 10)
    coords = []
    
    # 5. Get the coordinates of the frame
    # 6. draw the rectangle lines   
    # 7. Crop the face coodinates from the actual image
    # 8. Call the genrate_data function to save the croped face to the given location 
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        coords = (x, y, w, h)
        if len(coords)==4:
            aoi = img[y:y+h-3, x:x+w-3]
            genrate_data(FolderName, ImageName, img_id, aoi)        
            
    cv2.imshow("FaceDetection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# destroy all the windows 
video_capture.release()
cv2.destroyAllWindows()
