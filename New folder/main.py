import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os
from PIL import Image, ImageTk
import KNN
import naive_bayes
import logisticregression

import numpy as np
import preprocess as pp
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # no tensor flow debug msg

# Set up neural network

model = tf.keras.models.load_model('models/my_model')

batch_size = 32
image_height = 180
image_width = 180
data_dir = "data/arcDataset"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=2710,
    image_size=(image_height, image_width),
    batch_size=batch_size,
)

class_names = train_ds.class_names

X, Y, class_labels = pp.vectorize("data/arcDatasetProc2",
                                  False)  # use 96 x 96 for demo


train_data, train_labels, test_data, test_labels = pp.data_split(X, Y, 0.8)


root = tk.Tk()
root.title("Architecture Classification")
frame = tk.Frame(root)
frame.pack()
label = tk.Label(frame, text="Architecture Classification")
label.pack(ipadx=10, ipady=10, side=tk.TOP)

# empty is white image for base
empty_img = ImageTk.PhotoImage(Image.open("empty.jpg"))

img_frame = tk.Frame(frame)
panel = tk.Label(img_frame, image=empty_img)
panel.pack()
img_frame.pack(side=tk.TOP, ipadx=10, ipady=10)

labels = tk.Label(frame, text="Model Predictions: ")

pred4 = tk.Label(frame, text="Neural Net: ", font=("Arial", 25))
pred4.pack()

pred1 = tk.Label(frame, text="KNN: ")
pred1.pack()
pred2 = tk.Label(frame, text="Naive Bayes: ")
pred2.pack()
pred3 = tk.Label(frame, text="Logistic Regression: ")
pred3.pack()

# input image from file system, display the image, and run
# the models to classify it


def UploadImage(event=None):
    filename = filedialog.askopenfilename()

    img_torch = tf.keras.utils.load_img(filename, target_size=(256, 256))
    img = Image.open(filename)

    # NN pred

    img_array = tf.keras.utils.img_to_array(img_torch)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    pred4.configure(text="Neural Network: {} with a {:.2f} percent confidence."
                    .format(class_names[np.argmax(score)], 100 * np.max(score)))

    # KNN prediction, use all data

    proc_img = pp.proc_single(filename, (96, 96))

    predictions = KNN.knn_classify(X, Y, proc_img, 5)
    print(predictions)
    pred1.configure(text=("KNN: " + class_labels[predictions[0]]))

    # Naive Bayes prediction
    predictions = naive_bayes.nbpredict_map(X, Y, proc_img)

    pred2.configure(text=("Naive Bayes: " + class_labels[predictions[0]]))

    # Logistic Regression prediction

    predictions = logisticregression.predict_LR(
        logisticregression.adagrad, X, Y, proc_img, 1.0, 100, load=True)

    pred3.configure(text=("Logistic Regression: " +
                    class_labels[predictions[0]]))

    # display image
    img = img.resize((600, 600))
    img2 = ImageTk.PhotoImage(img)

    panel.configure(image=img2)
    panel.image = img2


# button for uploading images

button1 = tk.Button(frame, text='Open Image',
                    command=UploadImage)
button1.pack(side=tk.TOP, ipadx=10, ipady=10)


root.mainloop()
