Input = input("Эта программа загружает файл нейросети, распознающей цифры \"neural.network\", и работает с ним, передавая нейросети картинки \"custom1.jpg\", \"custom2.jpg\" ... \"custom10.jpg\" из папки \"data\". Чтобы просмотреть ответ нейросети на другую картинку, закройте окно с нынешней. Вы можете изменить картинки и добавлять новые. Чтобы завершить программу, введите \"exit\" на вопрос \"Картинка: data\custom\".\nНажмине Enter для продолжения...")

import numpy as np
import matplotlib.pyplot as plt

try:
    weightsFile = open("neural.network", "r")
except IOError:
    print("Откройте сперва файл \"save.py\", чтобы получить файл нейросети.")
    from time import sleep
    sleep(5)
    exit(1)
data = weightsFile.read().split("\nsplitdata\n")

weights_input_to_hidden = data[1].split("\n")
weights_hidden_to_output = data[3].split("\n")
bias_input_to_hidden = list(map(float, data[0].split("\n")))
bias_hidden_to_output = list(map(float, data[2].split("\n")))

del data
weightsFile.close()

for i in range(len(weights_input_to_hidden)):
    weights_input_to_hidden[i] = weights_input_to_hidden[i].split()
    weights_input_to_hidden[i] = list(map(float, weights_input_to_hidden[i]))
for i in range(len(weights_hidden_to_output)):
    weights_hidden_to_output[i] = weights_hidden_to_output[i].split(" ")
    weights_hidden_to_output[i] = list(map(float, weights_hidden_to_output[i]))

hidden_num = len(weights_input_to_hidden)

weights_input_to_hidden = np.reshape(weights_input_to_hidden, (hidden_num, 784))
weights_hidden_to_output = np.reshape(weights_hidden_to_output, (10, hidden_num))
bias_input_to_hidden = np.reshape(bias_input_to_hidden, (hidden_num, 1))
bias_hidden_to_output = np.reshape(bias_hidden_to_output, (10, 1))

picNum = ""

def nextPicture():
    global picNum
    picNum = input("Картинка: data\custom")
    if picNum.lower() == "exit":
        exit()

nextPicture()

while True:
    try:
        test_image = plt.imread(f"data\custom{picNum}.jpg", format="jpeg")
    except FileNotFoundError:
        print("Введите число так, чтобы получилось имя нужного файла и нажмите Enter.")
        nextPicture()
        continue

    gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
    test_image = 1 - (gray(test_image).astype("float32") / 255)

    test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))


    image = np.reshape(test_image, (-1, 1))
    hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
    hidden = 1 / (1 + np.exp(-hidden_raw))
    output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
    output = 1 / (1 + np.exp(-output_raw))

    plt.imshow(test_image.reshape(28, 28), cmap="Greys")
    plt.title(f"Ответ: {output.argmax()}")
    plt.show()
    nextPicture()
