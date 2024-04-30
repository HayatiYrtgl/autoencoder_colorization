import numpy as np
from keras.models import Sequential
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.optimizers import Adam

# load data
x_gray = np.load("../DATASET/land/gray_x.npy")
x_color = np.load("../DATASET/land/color_x.npy")

print(x_color.shape, x_gray.shape)

x_train_gray = x_gray[:5500]
x_train_color = x_color[:5500]

x_test_color = x_color[5500:]
x_test_gray = x_gray[5500:]

model = Sequential()

# encoder
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(128, 128, 1)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", strides=2, activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", strides=2, activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", strides=2, activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))

# decoder
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Dropout(0.1))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Dropout(0.1))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Dropout(0.1))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(3, (3, 3), activation="relu", padding="same"))
model.add(UpSampling2D((2, 2)))

# summary
model.compile(loss="mse", optimizer=Adam(learning_rate=0.0001), metrics=["acc"])

model.summary()

# chekpoint
cp = ModelCheckpoint(filepath="../models/colorizer", monitor="acc", save_best_only=True)

model.fit(x_train_gray, x_train_color, epochs=50, batch_size=32, validation_data=(x_test_gray, x_test_color),
          callbacks=[cp])

