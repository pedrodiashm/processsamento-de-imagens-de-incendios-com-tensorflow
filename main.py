import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image, UnidentifiedImageError
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Caminho principal do dataset
DATA_DIR = "abdelghaniaaba/wildfire-prediction-dataset/versions/1/"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42


def fix_corrupted_images(folder):
    folder = pathlib.Path(folder)
    fixed = 0
    removed = 0
    for path in folder.rglob("*.*"):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        try:
            # Tenta decodificar usando o TensorFlow
            img_raw = tf.io.read_file(str(path))
            _ = tf.image.decode_jpeg(img_raw, channels=3)
        except Exception:
            try:
                # Tenta corrigir regravando com Pillow
                with Image.open(path) as img:
                    rgb = img.convert("RGB")
                    rgb.save(path, "JPEG", quality=95)
                    fixed += 1
                    print(f"[✓] Corrigida: {path}")
            except (UnidentifiedImageError, OSError):
                # Se nem o Pillow conseguir → deleta
                print(f"[✗] Removida (irrecuperável): {path}")
                path.unlink(missing_ok=True)
                removed += 1
    print(f"\nResumo para {folder} → Corrigidas: {fixed}, Removidas: {removed}\n")


# Rodar para cada subpasta
for subdir in ["train", "valid", "test"]:
    fix_corrupted_images(os.path.join(DATA_DIR, subdir))


# ---------------------- PREPARAR DADOS ----------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "valid"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Normalizar imagens [0, 1]
normalization_layer = layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Otimizar pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---------------------- EXPLORAR COM PILLOW ----------------------
# Mostrar uma imagem exemplo
# for imgs, labels in train_ds.take(1):
#     img_array = imgs[0].numpy()
#     label = int(labels[0])
#     Image.fromarray((img_array * 255).astype(np.uint8)).show(title=f"Classe: {class_names[label]}")
#     break


# ---------------------- DEFINIR MODELO CNN ----------------------
# CNN construída do zero
def build_cnn(input_shape=(128, 128, 3)):
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


model = build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
model.summary()

# ---------------------- COMPILAR E TREINAR ----------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)

history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

# ---------------------- AVALIAÇÃO ----------------------
print("\n[+] Avaliação no conjunto de teste:")
results = model.evaluate(test_ds)
print("Loss, Accuracy, Precision, Recall:", results)

# Predições
y_true = []
y_pred = []
for x, y in test_ds:
    preds = model.predict(x)
    y_true.extend(y.numpy().astype(int))
    y_pred.extend((preds.ravel() > 0.5).astype(int))

print("\n[+] Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)


# ---------------------- GRÁFICOS ----------------------
def plot_training(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Treino")
    plt.plot(history.history["val_loss"], label="Validação")
    plt.title("Função de perda")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Treino")
    plt.plot(history.history["val_accuracy"], label="Validação")
    plt.title("Acurácia")
    plt.legend()
    plt.show()


plot_training(history)

# ---------------------- SALVAR MODELO ----------------------
os.makedirs("results", exist_ok=True)
model.save("results/wildfire_cnn.h5")
print("Modelo salvo em results/wildfire_cnn.h5")
