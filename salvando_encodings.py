import numpy as np

def save_data(file_path, data):
    with open(file_path, 'a') as file:
        for embedding, name in data:
            # Converter os valores de embedding em uma string
            embedding_str = ' '.join(map(str, embedding))

            # Escrever a linha no arquivo
            file.write(f"{embedding_str} {name}\n")

# Exemplo de uso
embeddings_to_save = [
    (np.array([0.1, 0.2, 0.3]), "Nome1"),
    (np.array([0.4, 0.5, 0.6]), "Nome2"),
    # Adicione mais pares de embeddings e nomes conforme necessário
]

# Especificar o caminho do arquivo
file_path = 'encodings.txt'

# Salvar os dados no arquivo
save_data(file_path, embeddings_to_save)