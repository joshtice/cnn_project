#! /usr/bin/env python3

"""

Acknowledgements
"""

from pathlib import Path

import cv2
from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
# from keras.applications.resnet50 import ResNet50, preprocess_input
# from keras.applications.xception import Xception, preprocess_input
from keras.applications import resnet50, xception
from keras.layers import Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

app = Flask(__name__)

# App configuration
app.config['SECRET_KEY'] = 'supersecretkeygoeshere'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dropzone configuration
dropzone = Dropzone(app)
app.config['DROPZONE_UPLOAD_MULTIPLE'] = False
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

# Uploads configuration
app.config['UPLOADED_PHOTOS_DEST'] = Path.cwd() / 'uploads'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)

# Load list of dog breed names
with open('utility_files/dog_names.pickle', 'rb') as f:
    dog_names = pickle.load(f)

# Load human face detector
face_cascade = cv2.CascadeClassifier(
    'utility_files/haarcascade_frontalface_alt.xml')

# Load resnet50 model for dog identifier
resnet50_model = resnet50.ResNet50(weights='imagenet')

# Load xception model for breed classifier
xception_model = Sequential()
xception_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
xception_model.add(Dense(133, activation='softmax'))
xception_model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])
xception_model.load_weights('utility_files/weights.best.Xception.hdf5')


def face_detector(img_path):
    """Detect human faces in a given image

    Parameters
    ----------
    img_path : str
        Path to the image

    Returns
    -------
    bool
        True if human face(s) detected in image
    """

    img = cv2.imread(img_path)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale_img)
    return len(faces) > 0


def path_to_tensor(img_path):
    """Converts image to tensor for input to deep learning model

    Parameters
    ----------
    img_path : str
        Path to image file to be converted to tensor

    Returns
    np.ndarray
        Tensor to serve as input for deep learning model
    """

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def dog_detector(img_path):
    """Returns "True" if a dog is detected in the image at img_path

    Parameters
    ----------
    img_path : str
        Path to the image of interest

    Returns
    -------
    bool
        "True if a dog is detected in the image
    """

    img = resnet50.preprocess_input(path_to_tensor(img_path))
    prediction = np.argmax(resnet50_model.predict(img))
    return ((prediction <= 268) & (prediction >= 151))


def predict_breed(img_path):
    """Predict the breed of a dog in a given image using a trained CNN

    Parameters
    ----------
    img_path : str
        The path to the image to be classified

    Returns
    -------
    str
        The predicted breed of the dog in the image
    """


    img = xception.preprocess_input(path_to_tensor(img_path))
    bottleneck_feature = xception.Xception(
        weights='imagenet',
        include_top=False).predict(img)
    predicted_vector = xception_model.predict(bottleneck_feature)
    prediction = dog_names[np.argmax(predicted_vector)]

    return prediction


def match_dog_breed(img_path):
    """Detect whether an image contains a dog or human, then guess breed

    Parameters
    ----------
    img_path : str
        Path to an image for classification
    """

    if face_detector(img_path):
        result = predict_breed(img_path)
    elif dog_detector(img_path):
        print("made it this far...")
        result = predict_breed(img_path)
    else:
        result = predict_breed(img_path)

    return result


@app.route('/', methods=['GET', 'POST'])
def index():

    # [file.unlink() for file in Path("./uploads").iterdir()]
    # session['file_urls'] = []
    # session['predictions'] = []

    # # list to hold uploaded image urls
    # file_urls = session['file_urls']
    # predictions = session['predictions']

    # # handle image upload from Dropszone
    # if request.method == 'POST':
    #     file_obj = request.files
    #     for f in file_obj:
    #         file = request.files.get(f)

    #         # save the file to the uploads folder
    #         filename = photos.save(
    #             file,
    #             name=file.filename
    #         )

    #         # append image urls
    #         file_urls.append(photos.url(filename))

    #         # predict dog breed based on image
    #         # img_path = str(Path.cwd() / "uploads" / file.filename)
    #         # prediction = match_dog_breed(img_path)
    #         # prediction = predict_breed(img_path)
    #         prediction = predict_breed("test_images/Brittany_02625.jpg")
    #         predictions.append(prediction)

    #     session['file_urls'] = file_urls
    #     session['predictions'] = predictions
    #     return "uploading..."

    result = predict_breed("test_images/Brittany_02625.jpg")

    # return dropzone template on GET request
    return render_template('index.html', result=result)


@app.route('/results')
def results():

    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))

    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    predictions = session['predictions']
    session.pop('file_urls', None)
    session.pop('predictions', None)

    return render_template('results.html', file_urls=file_urls, predictions=predictions)

