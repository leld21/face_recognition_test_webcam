#import face_recognition
import dlib
import cv2
import os,sys
import numpy as np
from typing import Union

import tkinter as tk
from tkinter.simpledialog import askstring
from PIL import Image, ImageTk

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

face_detector = dlib.get_frontal_face_detector()
file_path = 'encodings_tensorrt.txt'
embeddings_to_save =[]

def load_trt_engine(engine_path):
    # Carregando o arquivo .trt engine
    with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    # Alocando buffers para entrada e saída com base no tamanho do modelo
    h_input, d_input, h_output, d_output, bindings = [], [], [], [], []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            h_input.append(host_mem)
            d_input.append(device_mem)
        else:
            h_output.append(host_mem)
            d_output.append(device_mem)

    return h_input, d_input, h_output, d_output, bindings

engine_path = 'mobilefacenet_engine.trt'
engine = load_trt_engine(engine_path)
context = engine.create_execution_context()


# Alocando buffers de entrada e saída fora da função predict_trt
h_input, d_input, h_output, d_output, bindings = allocate_buffers(engine)

def predict_trt(input_data):
    # Copiando dados de entrada para o buffer alocado na GPU
    np.copyto(h_input[0], input_data.ravel())
    cuda.memcpy_htod(d_input[0], h_input[0])

    context.execute(batch_size=1, bindings=bindings)

    # Copiando os resultados de volta para a CPU
    cuda.memcpy_dtoh(h_output[0], d_output[0])

    # Retornando os resultados
    return h_output[0].reshape(engine.get_binding_shape(1))

def save_data_single(file_path, data):
    with open(file_path, 'a') as file:
            embedding_str = ' '.join(map(str, data[0]))
            file.write(f"{embedding_str} {data[1]}\n")

def save_data(file_path, data):
    with open(file_path, 'a') as file:
        for encoding, name in data:
            # Converter os valores de embedding em uma string
            encoding_str = ' '.join(map(str, encoding))

            # Escrever a linha no arquivo
            file.write(f"{encoding_str} {name}\n")

def load_data(file_path):
    embeddings_list = []

    with open(file_path, 'r') as file:
        for line in file:
            # Divida a linha em valores (embedding e nome)
            values = line.strip().split()

            # Primeiros 128 valores são floats (embedding)
            embedding = np.array([float(value) for value in values[:256]])

            # O restante da linha é o nome da pessoa
            name = ' '.join(values[256:])

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
                embedding_copy = np.copy(encoding)

                cv2.imwrite(f'faces/{nome}.png', image)
                self.tupla_encodings_nome.append((embedding_copy,nome))
    def encode_faces(self):
        for image in os.listdir('faces'):
            #imagem de fato da pessoa
            face_image = extract_faces('faces/' + image)
            #caracteristica ( embeddings ) dos rostos encontrados.
            face_encoding = predict_trt(face_image)[0]
            embedding_copy = np.copy(face_encoding)
            nome_pessoa = image[:-4]
            print(len(face_encoding))
            print(nome_pessoa)
            embeddings_to_save.append((embedding_copy,nome_pessoa))
        
    def run_recognition(self):
        # pega a primeira camera disponivel
        video_capture = cv2.VideoCapture(0)
        #video_capture = cv2.VideoCapture('http://192.168.0.6:4747/video')
	
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Inicializar a interface gráfica
        root = tk.Tk()

        root.attributes('-fullscreen', True)
        root.bind('<f>', lambda event: root.attributes('-fullscreen', not root.attributes('-fullscreen')))

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Criar um rótulo para exibir a imagem
        label = tk.Label(root)

        #trocar 1280 e 720 pela resolucao da imagem pega pela webcam.
        label.grid(row=0, column=0, padx=(screen_width - 1600) // 2)
        #label.grid(row=0, column=0)

        # Criar um botão na interface gráfica
        #botao = tk.Button(root, text="Adicionar Seu Rosto",height=3, width=15)
        botao = tk.Button(root, text="Adicionar Seu Rosto", height=2, width=18, bg="red", fg="yellow")

        botao.config(font=("Arial", 14))
        botao.grid(row=1, column=0, pady=10)

        root.bind('<Return>', lambda event=None: self.adicionar_nome(small_frame, self.face_encodings[0]))
        botao["command"] = lambda: self.adicionar_nome(small_frame, self.face_encodings[0])

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
                    self.face_encodings.append(predict_trt(face_image)[0])
                #self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                    
                self.face_names = []
                for face_encoding in self.face_encodings:
                    name = 'Desconhecido'
                    #confidence = 'Desconhecido'

                    #array com as distancias de cada rosto conhecido com o do frame atual
                    face_distances = [findCosineDistance(known_encoding, face_encoding) for known_encoding, _ in self.tupla_encodings_nome]
                    
                    #determina o elemento do array com menor distancia
                    best_match_index = np.argmin(face_distances)
                    #print(face_distances)

                    #verifica se o elemento é menor que o threshold do modelo
                    if face_distances[best_match_index] < 0.45:
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

            frame_stretched = cv2.resize(frame, (1600, 900))
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
