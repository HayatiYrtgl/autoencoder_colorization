from keras.models import load_model
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import numpy as np
import matplotlib.pyplot as plt
import glob

def colorize(test_set, model_path, target_size):
    model = load_model(model_path)
    # Görüntüyü yükle
    for i in test_set[:100]:
      img = load_img(i, target_size=target_size, color_mode="grayscale")

      # Görüntüyü modelin giriş formatına dönüştür
      img_array = img_to_array(img)
      img_array = np.expand_dims(img_array, axis=0)
      img_array /= 255

      # Tahmini yap
      predicted = model.predict(img_array)
      predicted = np.squeeze(predicted)
      predicted_img = array_to_img(predicted)

      # Orijinal ve tahmin edilen görüntüleri göster
      plt.figure(figsize=(10, 5))

      # Orijinal siyah-beyaz görüntü
      plt.subplot(1, 2, 1)
      plt.imshow(img, cmap='gray')
      plt.title('Orijinal Siyah-Beyaz')
      plt.axis('off')

      # Tahmin edilen renklendirilmiş görüntü
      plt.subplot(1, 2, 2)
      plt.imshow(predicted_img)
      plt.title('Tahmin Edilen Renklendirilmiş')
      plt.axis('off')

      plt.show()

colorize(glob.glob("../custom/*.*"), target_size=(128, 128), model_path="../models/colorizer")