# autoencoder_colorization
Colorizes grayscale images using a loaded model and displays original and predicted colorized versions.

This code block is for training a convolutional autoencoder model to colorize grayscale images. Let's break down the code and analyze it:

1. **Importing Libraries**: The necessary libraries such as NumPy, Keras layers, optimizers, and callbacks are imported.

2. **Loading Data**: Grayscale and corresponding color images are loaded from numpy files. The grayscale images are stored in `x_gray` and the color images in `x_color`.

3. **Data Preparation**: The loaded data is split into training and testing sets.

4. **Model Definition**:
   - The model architecture is defined using the Sequential API.
   - The encoder part consists of several Conv2D layers with increasing numbers of filters and varying strides. These layers are responsible for extracting features from the input grayscale images.
   - The decoder part mirrors the encoder architecture in reverse, aiming to reconstruct the colorized image from the extracted features. Dropout layers are included to prevent overfitting.
   - The final layer uses Conv2D with 3 filters (representing RGB channels) to produce the colorized output image.

5. **Model Compilation**: The model is compiled using Mean Squared Error (MSE) as the loss function and Adam optimizer with a learning rate of 0.0001. The accuracy metric is also specified.

6. **Model Summary**: The summary of the model architecture is printed, displaying the layers, output shapes, and number of parameters.

7. **Model Training**:
   - The model is trained using the `fit` method on the training data (`x_train_gray` and `x_train_color`).
   - It is trained for 50 epochs with a batch size of 32.
   - Validation data (`x_test_gray` and `x_test_color`) is provided for evaluating the model's performance during training.
   - ModelCheckpoint callback is used to save the best model based on validation accuracy.

Overall, this code block defines, compiles, trains, and evaluates a convolutional autoencoder model for colorizing grayscale images using Keras. The model architecture consists of encoder and decoder parts, and it is trained to minimize the reconstruction error between the colorized images and the ground truth color images.

This code block defines a function `process` that preprocesses image data for a colorization task. Let's analyze it step by step:

1. **Importing Libraries**: The necessary libraries such as `load_img` and `img_to_array` from Keras, NumPy, and glob are imported.

2. **Function Definition**:
   - `process` function is defined, which takes two arguments: `path` (path to the directory containing the images) and `target_size` (desired size of the images after resizing).
   - Inside the function, empty lists `x` and `y` are created to store grayscale and corresponding color images, respectively.

3. **Processing Images**:
   - The function iterates through each image file in the specified directory using `glob.glob(path)`.
   - For each image file, it loads the color image (`img_color`) and grayscale image (`img_gray`) using the `load_img` function from Keras. Both images are resized to the specified `target_size` and converted to NumPy arrays using `img_to_array`.
   - The pixel values of both grayscale and color images are normalized to the range [0, 1] by dividing by 255.
   - The normalized images are appended to the `x` and `y` lists, respectively.

4. **Converting to NumPy Arrays**:
   - After processing all images, the lists `x` and `y` are converted to NumPy arrays using `np.array`.

5. **Saving Processed Data**:
   - Finally, the processed grayscale images (`x`) and corresponding color images (`y`) are saved as NumPy arrays using `np.save`.

6. **Function Call**:
   - The `process` function is called with the specified `path` and `target_size`.

Overall, this function preprocesses color images by converting them to grayscale, resizing them, and normalizing their pixel values. It then saves the processed grayscale and color images as NumPy arrays, which can be used for training a colorization model.

This code defines a function named `colorize` that takes a list of image paths (`test_set`), the path to a trained model (`model_path`), and the `target_size` of the images. It then loads each grayscale image from the test set, passes it through the loaded model to predict the colorized version, and displays the original grayscale image along with the predicted colorized image using matplotlib.

Let's break down the code:

1. **Importing Libraries**: The necessary libraries such as `load_model`, `img_to_array`, `array_to_img`, `load_img`, `numpy`, `matplotlib.pyplot`, and `glob` are imported.

2. **Function Definition**:
   - The `colorize` function is defined with parameters `test_set`, `model_path`, and `target_size`.
   - Inside the function, the model is loaded using `load_model` from the specified `model_path`.

3. **Colorization Loop**:
   - The function iterates through each image path in the `test_set`.
   - For each image, it loads the grayscale image using `load_img`, converts it to a NumPy array, expands its dimensions to match the input format expected by the model, and normalizes the pixel values.
   - The normalized grayscale image is passed through the model to predict the colorized version.
   - The predicted colorized image is converted back to an image format using `array_to_img`.

4. **Displaying Results**:
   - For each grayscale image, the function plots two subplots using matplotlib.
   - The first subplot displays the original grayscale image, and the second subplot displays the predicted colorized image.
   - Finally, `plt.show()` is called to display the plot.

5. **Function Call**:
   - The `colorize` function is called with the specified parameters: test set of images (obtained using `glob.glob`), `target_size`, and the path to the trained model.

This function can be used to visualize the colorized versions of grayscale images using the trained model. However, it seems like there might be an issue with the input images' paths as `glob.glob("../custom/*.*")` may not be pointing to the correct directory. You might need to adjust the path according to your directory structure.
