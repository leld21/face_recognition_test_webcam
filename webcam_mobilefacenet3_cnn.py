#import face_recognition
import dlib
import cv2
import os,sys
import numpy as np
import tensorflow as tf
from typing import Union

#face_detector = dlib.get_frontal_face_detector()
face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

def findCosineDistance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def rect_to_bb(rect):
    x = rect.rect.left()
    y = rect.rect.top()
    w = rect.rect.width()
    h = rect.rect.height()

    return x, y, w, h

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return x - x_off, x + width + x_off, y - y_off, y + height + y_off

def get_face_locations(imagem):
    
    image_gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    
    face_coord = face_detector(image_gray, 1)
    
    return face_coord
    
def extract_faces(image_path, resolution=(112, 112)):

    image = cv2.imread(image_path)

    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # detector de rosto
    face_coord = face_detector(image_gray, 1)

    if len(face_coord) > 0:
        (x, y, w, h) = rect_to_bb(face_coord[0])
        x1, x2, y1, y2 = apply_offsets((x, y, w, h), (10, 10))

        # cortar rosto
        image_face = image[y1:y2, x1:x2, :]

        image_face = np.asarray(image_face, dtype="float32")
        if image_face.shape[0] != 0 and image_face.shape[1] and image_face.shape[2] != 0:

            image_resized = cv2.resize(image_face, resolution, interpolation = cv2.INTER_AREA)

            image_resized /= 255
            
            image_resized = np.expand_dims(image_resized, axis=0)
            
            return image_resized

    else:
        return None

def extract_face2(image , face_coord, resolution=(112, 112),):
    if len(face_coord) > 0:
        (x, y, w, h) = rect_to_bb(face_coord[0])
        x1, x2, y1, y2 = apply_offsets((x, y, w, h), (10, 10))
        # cortar rosto
        image_face = image[y1:y2, x1:x2, :]

        image_face = np.asarray(image_face, dtype="float32")
        if image_face.shape[0] != 0 and image_face.shape[1] and image_face.shape[2] != 0:
            #prepara a imagem para ser aceita pelo modelo.

            image_resized = cv2.resize(image_face, resolution, interpolation = cv2.INTER_AREA)

            image_resized /= 255
            
            image_resized = np.expand_dims(image_resized, axis=0)
            
            return image_resized
    else:
        return None
    
class FaceRecognition:
    path = "keras_se_mobile_facenet_emore_triplet_basic_agedb_30_epoch_100_0.958333.h5"
    model = tf.keras.models.load_model(path)
    
    face_locations = [] 
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    
    def __init__(self):
        self.encode_faces()
        
    def encode_faces(self):
        for image in os.listdir('faces'):
            #imagem de fato da pessoa
            face_image = extract_faces('faces/' + image)
            #caracteristica ( embeddings ) dos rostos encontrados.
            face_encoding = self.model.predict(face_image, verbose=0)[0]
            
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image[:-4])
    
        print(self.known_face_names)
        
    def run_recognition(self):
        # pega a primeira camera disponivel
        #video_capture = cv2.VideoCapture(0)
        video_capture = cv2.VideoCapture('http://192.168.0.6:4747/video')

        if not video_capture.isOpened():
            sys.exit('camera nao encontrada')
            
        while True:
            ret, frame = video_capture.read()
            
            #Processa 1 vez a cada 2 frames
            if (self.process_current_frame):
                small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                #Encontra todos rostos no frame atual
                
                #self.face_locations = face_recognition.face_locations(rgb_small_frame)
                
                self.face_locations = get_face_locations(rgb_small_frame)
                
                face_image = extract_face2(rgb_small_frame, self.face_locations)
                
                self.face_encodings = []
                
                if face_image is not None:
                    self.face_encodings.append(self.model.predict(face_image, verbose=0)[0])
                
                #self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                    
                self.face_names = []
                for face_encoding in self.face_encodings:
                    name = 'Desconhecido'
                    #confidence = 'Desconhecido'

                    #array com as distancias de cada rosto conhecido com o do frame atual
                    face_distances = [findCosineDistance(known_encoding, face_encoding) for known_encoding in self.known_face_encodings]
                    
                    #determina o elemento do array com menor distancia
                    best_match_index = np.argmin(face_distances)
                    #print(face_distances)

                    #verifica se o elemento é menor que o threshold do modelo
                    if face_distances[best_match_index] < 0.6:
                        name = self.known_face_names[best_match_index]
                    
                    self.face_names.append(f'{name}')
                    
            self.process_current_frame = not self.process_current_frame
            
            #Mostrar anotacoes na imagem
            if len(self.face_locations) > 0:
                for face_location, name in zip(self.face_locations, self.face_names):
                    top = face_location.rect.top()*4
                    right = face_location.rect.right()*4
                    bottom = face_location.rect.height()*4
                    left = face_location.rect.width()*4
                        
                    cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255), 2)
                    cv2.rectangle(frame, (left,bottom - 35), (right,bottom), (0,0,255), -1)  
                    cv2.putText(frame, name, (left + 6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                
            cv2.imshow('Teste Face Recognition', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
                
if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()