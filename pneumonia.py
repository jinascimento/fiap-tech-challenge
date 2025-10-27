import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "xray/train"
val_dir = "xray/val"
test_dir = "xray/test"

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Gera imagens de treino (com Data Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normaliza os pixels para 0-1
    rotation_range=20,  # Rotaciona aleatoriamente
    width_shift_range=0.1,  # Desloca horizontalmente
    height_shift_range=0.1,  # Desloca verticalmente
    shear_range=0.1,  # "Inclina" a imagem
    zoom_range=0.1,  # Zoom aleatório
    horizontal_flip=True,  # Inverte horizontalmente
    fill_mode="nearest",
)

# Gera imagens para validação/teste (apenas normalização)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Carrega dados dos diretórios, iremos usar o class_mode como binário
# por conta da natureza binária do problema (Normal vs Pneumonia)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,  # Importante para avaliação
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

# Criando o modelo com N camadas
model = Sequential(
    [
        # Camada 1
        Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        ),
        MaxPooling2D((2, 2)),
        # Camada 2
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        # Camada 3
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        # Camada 4
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        # Camadas de classificação
        Flatten(),
        Dropout(0.5),  # Regularização
        Dense(512, activation="relu"),
        Dense(
            1, activation="sigmoid"
        ),  # 1 neurônio de saída para classificação binária
    ]
)

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)

# Para este dataset, os pesos são aproximadamente os seguintes:
# 'NORMAL' (0) é a minoria, 'PNEUMONIA' (1) é a maioria.
# O Peso para classe 0 (Normal) deve ser maior.
# Ex:
# total_amostras = 5216 (treino)
# total_normal = 1341
# total_pneumonia = 3875
# peso_para_0 = (1 / total_normal) * (total_amostras / 2.0) ~ 1.94
# peso_para_1 = (1 / total_pneumonia) * (total_amostras / 2.0) ~ 0.67
class_weights = {0: 1.94, 1: 0.67}  # {classe_normal: peso, classe_pneumonia: peso}

print(f"Índices das classes: {train_generator.class_indices}")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    class_weight=class_weights,
)


# Avaliação básica
results = model.evaluate(test_generator)

print(f"Loss no Teste: {results[0]}")
print(f"Acurácia no Teste: {results[1]}")
print(f"Precisão no Teste: {results[2]}")
print(f"Recall no Teste: {results[3]}")

# Matriz de Confusão e Relatório de Classificação

# Obter as previsões
Y_pred = model.predict(test_generator)
y_pred = np.round(Y_pred).flatten().astype(int)  # Arredonda (0 ou 1)

# Obter os rótulos verdadeiros
y_true = test_generator.classes

# Imprimir Relatório
print("\nRelatório de Classificação:")
print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

# Imprimir Matriz de Confusão
print("\nMatriz de Confusão:")
cm = confusion_matrix(y_true, y_pred)
print(cm)
