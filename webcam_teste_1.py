import face_recognition
import cv2
import os,sys
import numpy as np
import math

def face_confidence (face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range *2)
    
    if face_distance > face_match_threshold:
        return str(round(linear_val * 100,2)) + '%'
    else: 
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'
    

class FaceRecognition:
    face_locations = [] 
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = 0
    
    def __init__(self):
        self.encode_faces()
        
    def encode_faces(self):
        for image in os.listdir('faces'):
            #imagem de fato da pessoa
            face_image = face_recognition.load_image_file(f'faces/{image}')
            #caracteristica ( embeddings ) dos rostos encontrados.
            face_encoding = face_recognition.face_encodings(face_image)[0]
            
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
    
        print(self.known_face_names)
        
    def run_recognition(self):
        # pega a primeira camera disponivel
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            sys.exit('camera nao encontrada')
            
        while True:
            self.process_current_frame = self.process_current_frame + 1
            ret, frame = video_capture.read()
            
            if (self.process_current_frame%2 == 0):
                small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame
                
                #Encontra todos rostos no frame atual
                
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                    
                self.face_names = []
                for face_encoding in self.face_encodings:
                    #compara o rosto atual com a lista de rostos conhecidos
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Desconhecido'
                    confidence = 'Desconhecido'

                    #?? determina o rosto com menor distancia ao atual
                    face_distances = face_recognition.face_distance(self. known_face_encodings, face_encoding)
                    best_match_index =  np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                    
                    self.face_names.append(f'{name} ({confidence})')
                    
            #self.process_current_frame = not self.process_current_frame
            
            #Mostrar anotacoes na imagem
            
            for(top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4 
                    
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