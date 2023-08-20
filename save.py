Input = input("Даная программа обучает на распознаыание цифр (от 0 до 9) и записывает в файл \"neural.network\" нейросеть. Обучение - штука сложная, даже для нейросети. Если вы согласны на обучение нейросети, то впишите \"Хорошо\" и нажмите Enter. ")
if Input.lower() != "хорошо":
    exit()

hidden_num = int(input("Способность нейросети запоминать? (Влияет на скорость и качество обучения, стандартное значение - 20): "))
epochs = int(input("Количество циклов обучения: "))

import numpy as np

def load_dataset():
    with np.load("data\mnist.npz") as f:
        x_train = f['x_train'].astype("float32") / 255

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

        y_train = f['y_train']

        y_train = np.eye(10)[y_train]

        return x_train, y_train

images, labels = load_dataset()

weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (hidden_num, 784))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, hidden_num))
bias_input_to_hidden = np.zeros((hidden_num, 1))
bias_hidden_to_output = np.zeros((10, 1))

e_loss = e_correct = 0
learning_rate = 0.01

for epoch in range(epochs):
    print(f"Цикл обучения №{epoch+1}/{epochs}")

    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        hidden = 1 / (1 + np.exp(-hidden_raw))
        
        output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
        output = 1 / (1 + np.exp(-output_raw))

        e_loss += 1 / len(output) * np.sum((output-label)**2, axis=0)
        e_correct += int(np.argmax(output) == np.argmax(label))

        delta_output = output - label
        weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
        bias_hidden_to_output += -learning_rate * delta_output

        delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
        weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
        bias_input_to_hidden += -learning_rate * delta_hidden
    
    print(f"Ошибка: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
    print(f"Процент правильных ответов: {round((e_correct / images.shape[0]) * 100, 3)}%")
    e_loss = e_correct = 0

Input = input("Нейросеть запишется в файл. Впишите \"Хорошо\" и нажмите Enter, если согласны. ")

if Input.lower() == "хорошо":
    weightsFile = open("neural.network", "w")
    for i in range(bias_input_to_hidden.shape[0]):
        print(str(round(bias_input_to_hidden[i][0], 4)), file=weightsFile, flush=True)
    print("splitdata", file=weightsFile, flush=True)
    for i in range(weights_input_to_hidden.shape[0]):
        for j in range(weights_input_to_hidden.shape[1]):
            print(str(round(weights_input_to_hidden[i][j], 4)), end=" " if j != weights_input_to_hidden.shape[1]-1 else "", file=weightsFile, flush=True)
        print("\n" if i != weights_input_to_hidden.shape[0]-1 else "", end="", file=weightsFile, flush=True)
    print("\nsplitdata", file=weightsFile, flush=True)
    for i in range(bias_hidden_to_output.shape[0]):
        print(str(round(bias_hidden_to_output[i][0], 4)), file=weightsFile, flush=True)
    print("splitdata", file=weightsFile, flush=True)
    for i in range(weights_hidden_to_output.shape[0]):
        for j in range(weights_hidden_to_output.shape[1]):
            print(str(round(weights_hidden_to_output[i][j], 4)), end=" " if j != weights_hidden_to_output.shape[1]-1 else "", file=weightsFile, flush=True)
        print("\n" if i != weights_hidden_to_output.shape[0]-1 else "", end="", file=weightsFile, flush=True)
