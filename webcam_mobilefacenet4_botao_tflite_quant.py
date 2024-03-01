#import face_recognition
import dlib
import cv2
import os,sys
import numpy as np
import tensorflow as tf
from typing import Union

import tkinter as tk
from tkinter.simpledialog import askstring
from PIL import Image, ImageTk

model_path = "mobilefacenetquant.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Obter detalhes dos tensores de entrada e saída
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

face_detector = dlib.get_frontal_face_detector()
file_path = 'encodings_tflite_quant.txt'
embeddings_to_save =[]

def save_data_single(file_path, data):
    with open(file_path, 'a') as file:
            embedding_str = ' '.join(map(str, data[0]))
            file.write(f"{embedding_str} {data[1]}\n")

def save_data(file_path, data):
    with open(file_path, 'a') as file:
        for embedding, name in data:
            # Converter os valores de embedding em uma string
            embedding_str = ' '.join(map(str, embedding))

            # Escrever a linha no arquivo
            file.write(f"{embedding_str} {name}\n")

def load_data(file_path):
    embeddings_list = []

    with open(file_path, 'r') as file:
        for line in file:
            # Divida a linha em valores (embedding e nome)
            values = line.strip().split()

            embedding = np.array([float(value) for value in values[:128]])

            # O restante da linha é o nome da pessoa
            name = ' '.join(values[128:])

            # Adicione o par (embedding, nome) à lista
            embeddings_list.append((embedding, name))

    return embeddings_list

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
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

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

def get_embedding(image):
    # Alimentar a imagem para o modelo
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Obter os resultados (embedding)
    embedding = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Converter o embedding para um array NumPy
    embedding = np.array(embedding)

    return embedding

class FaceRecognition:
    face_locations = [] 
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    tupla_encodings_nome = []
    process_current_frame = True
    def __init__(self):
        #self.encode_faces()
        #save_data(file_path, embeddings_to_save)
        self.tupla_encodings_nome = load_data(file_path)

    def adicionar_nome(self,image, encoding):
            # Diálogo para inserir o nome da pessoa
            nome = askstring("Adicionar Nome", "Digite o seu Nome e Sobrenome:")

            while nome == '':
                nome = askstring("Adicionar Nome", "Nome invalido. Digite o seu Nome e Sobrenome:")
            
            # Se o usuário pressionar cancelar, o nome será None
            if (nome is not None and encoding is not None):
                original_nome = nome
                index = 1

                while any(existing_nome == nome for _, existing_nome in self.tupla_encodings_nome):
                    nome = f"{original_nome} {index}"
                    index += 1
        
                save_data_single(file_path,(encoding,nome))

                cv2.imwrite(f'faces/{nome}.png', image)

                self.tupla_encodings_nome.append((encoding,nome))

    def encode_faces(self):
        for image in os.listdir('faces'):
            #imagem de fato da pessoa
            face_image = extract_faces('faces/' + image)
            #caracteristica ( embeddings ) dos rostos encontrados.
            face_encoding = get_embedding(face_image)
            
            self.known_face_encodings.append(face_encoding)
            nome_pessoa = image[:-4]
            self.known_face_names.append(nome_pessoa)
            print(len(face_encoding))
            print(nome_pessoa)
            embeddings_to_save.append((face_encoding,nome_pessoa))
        print(self.known_face_names)
        
    def run_recognition(self):
        # pega a primeira camera disponivel
        video_capture = cv2.VideoCapture(0)
        #video_capture = cv2.VideoCapture('http://192.168.43.70:4747/video')

        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Inicializar a interface gráfica
        root = tk.Tk()

        root.attributes('-fullscreen', True)
        root.bind('<f>', lambda event: root.attributes('-fullscreen', not root.attributes('-fullscreen')))

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Criar um rótulo para exibir a imagem
        label = tk.Label(root)
        
        #trocar 1280 e 720 pela resolucao da imagem pega pela webcam.
        label.grid(row=0, column=0, padx=(screen_width - 1280) // 2)
        #label.grid(row=0, column=0)

        # Criar um botão na interface gráfica
        #botao = tk.Button(root, text="Adicionar Seu Rosto",height=3, width=15)
        botao = tk.Button(root, text="Adicionar Seu Rosto", height=2, width=18, bg="red", fg="yellow")

        botao.config(font=("Arial", 14))
        botao.grid(row=1, column=0, pady=10)

        if not video_capture.isOpened():
            sys.exit('camera nao encontrada')
            
        while True:
            ret, frame = video_capture.read()
            
            #Processa 1 vez a cada 2 frames
            if (self.process_current_frame):
                small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                #rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                self.face_locations = get_face_locations(small_frame)
                
                face_image = extract_face2(small_frame, self.face_locations)
                
                self.face_encodings = []
                
                if face_image is not None:
                    self.face_encodings.append(get_embedding(face_image))
                    root.bind('<Return>', lambda event=None: self.adicionar_nome(small_frame, self.face_encodings[0]))
                    botao["command"] = lambda: self.adicionar_nome(small_frame, self.face_encodings[0])
                    
                #self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                    
                self.face_names = []
                for face_encoding in self.face_encodings:
                    name = 'Desconhecido'
                    #confidence = 'Desconhecido'

                    #array com as distancias de cada rosto conhecido com o do frame atual
                    face_distances = [findCosineDistance(known_encoding, face_encoding) for known_encoding, _ in self.tupla_encodings_nome]
                    
                    #determina o elemento do array com menor distancia
                    best_match_index = np.argmin(face_distances)
                    print(face_distances)

                    #verifica se o elemento é menor que o threshold do modelo
                    if face_distances[best_match_index] < 0.3:
                        #name = self.known_face_names[best_match_index]
                        name = self.tupla_encodings_nome[best_match_index][1]
                    
                    self.face_names.append(f'{name}')
                    
            self.process_current_frame = not self.process_current_frame
            
            #Mostrar anotacoes na imagem
            if len(self.face_locations) > 0:
                for face_location, name in zip(self.face_locations, self.face_names):
                    top = face_location.top()*2
                    right = face_location.right()*2
                    bottom = face_location.bottom()*2
                    left = face_location.left()*2
                    
                    if name != 'Desconhecido':
                        text = 'Bem vindo, ' + name + '!'
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]

                        text_x = (frame.shape[1] - text_size[0]) // 2
                        text_y = 50  
                        
                        # Define a cor amarela (BGR)
                        color = (0, 255, 255)
                        
                        font_scale = 1.2
                        
                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, 2)
                    else:
                        text = 'Ola! Voce ainda nao esta cadastrado'
                        text2 = 'Clique no botao abaixo ou aperte Enter'
                        text3 = 'Quando o seu rosto estiver enquadrado'
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]

                        text_x = (frame.shape[1] - text_size[0]) // 2
                        text_y = 50  
                        
                        # Define a cor amarela (BGR)
                        color = (0, 255, 255)
                        
                        cv2.putText(frame, text, (text_x+60, text_y), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                        cv2.putText(frame, text2, (text_x+60, text_y+50), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
                        cv2.putText(frame, text3, (text_x+60, text_y+100), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
                    #a seguir, logica para colocar o nome no meio do retangulo do rosto.
                    name_text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)[0][0]
                    space_difference = left + ((right - left) // 2)
                    space_difference2 = (name_text_size // 2) + 2

                    cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255), 2)
                    cv2.rectangle(frame, (space_difference - space_difference2,bottom + 35), ((space_difference+space_difference2),bottom), (0,0,255), -1)  
                    cv2.putText(frame, name, (space_difference - space_difference2 , bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)

            frame_stretched = cv2.resize(frame, (1280, 720))
            #cv2.imshow('Face Recognition', frame_stretched)

            frame_rgb = cv2.cvtColor(frame_stretched, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            label.configure(image=img_tk)

            root.update()
            
            if cv2.waitKey(1) == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
                
if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()