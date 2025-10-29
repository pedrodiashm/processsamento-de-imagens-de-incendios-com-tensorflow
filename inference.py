from PIL import Image
import numpy as np
import tensorflow as tf

IMG_SIZE = (128, 128)
CLASS_NAMES = ["nowildfire", "wildfire"]

model = tf.keras.models.load_model("results/wildfire_cnn.h5")


def predict_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = model.predict(arr)[0][0]
    label = CLASS_NAMES[int(prob > 0.5)]
    print(f"Predição: {label} (prob={prob:.3f})")


# Exemplo
if __name__ == "__main__":
    predict_image(
        "abdelghaniaaba/wildfire-prediction-dataset/versions/1/test/nowildfire/-73.535,45.480806.jpg"
    )
