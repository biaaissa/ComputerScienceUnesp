import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
import numpy as np

# Carregar o conjunto de dados MNIST
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Normalizar os dados
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

# Construir o modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f'Acurácia no conjunto de teste: {test_acc}')

# Plotar a precisão de treinamento e validação ao longo do treinamento
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Escolher uma imagem de teste aleatória
random_index = np.random.randint(0, len(x_test))
test_image = x_test[random_index]
test_label = y_test[random_index]

# Fazer a predição
prediction = model.predict(np.expand_dims(test_image, axis=0))
predicted_label = np.argmax(prediction)

# Plotar a imagem e imprimir o número predito
plt.imshow(test_image.squeeze(), cmap='gray')
plt.title(f'Número real: {test_label}, Número predito: {predicted_label}')
plt.axis('off')
plt.show()
