import cv2
import tkinter as tk
from tkinter.simpledialog import askstring
from PIL import Image, ImageTk

# Função chamada quando o botão é clicado
def adicionar_nome():
    # Diálogo para inserir o nome da pessoa
    nome = askstring("Adicionar Nome", "Digite o seu Nome e Sobrenome:")

    while nome == '':
        nome = askstring("Adicionar Nome", "Nome invalido. Digite o seu Nome e Sobrenome:")
    
    # Se o usuário pressionar cancelar, o nome será None
    if nome is not None:
        # Salvar o nome ou realizar qualquer outra ação desejada
        print(f"Nome da pessoa: {nome}")

# Inicializar a interface gráfica
root = tk.Tk()

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Criar um rótulo para exibir a imagem
label = tk.Label(root)
label.grid(row=0, column=0)

# Criar um botão na interface gráfica
botao = tk.Button(root, text="Adicionar Nome", command=adicionar_nome)
botao.grid(row=1, column=0, pady=10)

while True:
    # Capturar um quadro da câmera
    ret, frame = cap.read()

    # Converter o frame para o formato RGB (tkinter usa RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Converter o frame para o formato PhotoImage
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    # Atualizar a imagem no rótulo
    label.configure(image=img_tk)
    label.img_tk = img_tk  # Manter uma referência para evitar a coleta de lixo

    # Esperar por eventos na interface gráfica (atualizar a interface)
    root.update()

    # Se a tecla 'q' for pressionada, sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
