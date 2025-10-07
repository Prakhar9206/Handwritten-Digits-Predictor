import numpy as np
from keras.models import load_model
from PIL import Image

print("loading model")
MODEL = load_model("digit_predictor.model.keras")
print("model loaded")


path = ""
while path != 'exit':
    path = input("enter number from 0 to 9999 or type 'exit' to quit:\n")
    if path == 'exit':
        print("quitting...")
        break
    else:
        img = Image.open(f"digit_images/{path}.png")
        img_array = np.array(img)

        image = img_array/255

        print(f"img_array shape = {img_array.shape}")
        print(f"image shape = {image.shape}")

        prediction = np.argmax(MODEL.predict(image.reshape(1,28,28)))
        print(f"Model predicts - {prediction}\n\n")
        img.show()