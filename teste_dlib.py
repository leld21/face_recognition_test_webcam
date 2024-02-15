import dlib
import cv2
import face_recognition
image_path = "faces/Arthur.jpg"

image = cv2.imread(image_path)

image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
face_detector = dlib.get_frontal_face_detector()
    
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
face_coord = face_detector(image_gray, 1)

face_location = face_recognition.face_locations(image_gray)

print(face_coord)

primeiro_elemento, segundo_elemento, terceiro_elemento, quarto_elemento = face_coord[0]
print(primeiro_elemento)
print(terceiro_elemento)

print(face_location)