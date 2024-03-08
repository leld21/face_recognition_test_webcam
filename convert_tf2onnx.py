import tensorflow as tf
import tf2onnx

# Carregar o modelo TensorFlow em formato .h5
model_path = 'keras_se_mobile_facenet_emore_triplet_basic_agedb_30_epoch_100_0.958333.h5'
model = tf.keras.models.load_model(model_path)

# Configurar o opset para 13
opset_version = 13

# Converter o modelo para ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=opset_version)
# Salvar o modelo ONNX
onnx_path = 'mobilefacenet.onnx'
with open(onnx_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())

print(f'Modelo convertido com sucesso para: {onnx_path}')
