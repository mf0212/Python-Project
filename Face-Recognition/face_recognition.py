import cv2
import dlib

#read the image

img = cv2.imread("cr7.png")

#convert img to grayscale: 3D->2D

gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

#dlib : Load predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#dlib: Load Face Recognition Detector
face_detector = dlib.get_frontal_face_detector()

#use detector to find face landmarks
faces = face_detector(gray)
#face include array of face
for face in faces:
    x1 = face.left()# left Point
    y1 = face.top()# top Point
    x2 = face.right()#right Point
    y2 = face.bottom()#bottom Point

    #Draw a rectangle
    cv2.rectangle(img = img, pt1=(x1,y1)
    ,pt2=(x2,y2),color=(0,255,0),thickness=3)

    face_features = predictor(image=gray, box=face,)
    #Loop through all 68 point
    for n in range(0,68):
        x = face_features.part(n).x
        y = face_features.part(n).y

        #Draw a circle
        cv2.circle(img = img, center=(x,y),radius = 2, color=(0,0,255),thickness = 2)
#show the image

cv2.imshow(winname="Face Recognition App", mat=img)

#wait for a key press to exit
cv2.waitKey(delay=0)

#Close all windows
cv2.destroyAllWindows()