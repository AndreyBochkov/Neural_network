Input = input("Эта программа загружает файл нейросети, распознающей цифры \"neural.network\", и работает с ним, передавая нейросети картинки \"custom1.jpg\", \"custom2.jpg\" ... \"custom5.jpg\" из папки \"data\". Чтобы просмотреть ответ нейросети на следующую картинку, закройте окно с нынешней. Вы можете изменить картинки.\nНажмине Enter для продолжения...")

import numpy as np
import matplotlib.pyplot as plt

try:
    weightsFile = open("neural.network", "r")
except IOError:
    print("Откройте сперва файл \"save.py\", чтобы получить файл нейросети.")
data = weightsFile.read().split("\nsplitdata\n")
picsNum = 5

weights_input_to_hidden = data[1].split("\n")
weights_hidden_to_output = data[3].split("\n")
bias_input_to_hidden = data[0].split("\n")
bias_hidden_to_output = data[2].split("\n")

del data
weightsFile.close()

for i in range(len(weights_input_to_hidden)):
    weights_input_to_hidden[i] = weights_input_to_hidden[i].split(" ")
    for j in range(len(weights_input_to_hidden[i])):
        weights_input_to_hidden[i][j] = float(weights_input_to_hidden[i][j])
for i in range(len(weights_hidden_to_output)):
    weights_hidden_to_output[i] = weights_hidden_to_output[i].split(" ")
    for j in range(len(weights_hidden_to_output[i])):
        weights_hidden_to_output[i][j] = float(weights_hidden_to_output[i][j])
for i in range(len(bias_input_to_hidden)):
    bias_input_to_hidden[i] = float(bias_input_to_hidden[i])
for i in range(len(bias_hidden_to_output)):
    bias_hidden_to_output[i] = float(bias_hidden_to_output[i])

hidden_num = len(weights_input_to_hidden)
weights_input_to_hidden = np.reshape(weights_input_to_hidden, (hidden_num, 784))
weights_hidden_to_output = np.reshape(weights_hidden_to_output, (10, hidden_num))
bias_input_to_hidden = np.reshape(bias_input_to_hidden, (hidden_num, 1))
bias_hidden_to_output = np.reshape(bias_hidden_to_output, (10, 1))

iteration = 1

while iteration <= picsNum:
    test_image = plt.imread(f"data\custom{iteration}.jpg", format="jpeg")

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
    iteration += 1
