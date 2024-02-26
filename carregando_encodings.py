import numpy as np

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
            print(name)
    return embeddings_list

# Exemplo de uso
file_path = 'encodings.txt'
data = load_data(file_path)

# Agora, a variável 'data' contém uma lista de tuplas, cada uma com um vetor de embedding e um nome associado.