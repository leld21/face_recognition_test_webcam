import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


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

# Exemplo de uso
if __name__ == '__main__':
    # Substitua 'sua_imagem.jpg' pelo caminho da sua imagem de entrada
    imagem_entrada = '/home/jetson-user/Desktop/face_recognition_test_webcam/faces/Leandro.jpg'

    # Carregando a imagem de entrada
    input_data = cv2.imread(imagem_entrada, cv2.IMREAD_COLOR)
    input_data = cv2.resize(input_data, (112, 112))  # Ajuste o tamanho conforme necessário

    # Normalizando a imagem de entrada, se necessário
    input_data = input_data.astype(np.float32) / 255.0

    # Realizando a inferência com TensorRT
    resultado = predict_trt(input_data)
    print("Resultado da inferência:", resultado)
    resultado2 = predict_trt(input_data)
    print("Resultado da inferência2:", resultado2)
    resultado3 = predict_trt(input_data)
    print("Resultado da inferência3:", resultado3)

