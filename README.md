# Dog Breed Classification App

## 1. Installation
The Dog Breed Classification App was developed using Python 3.7. The app also requires several packages not included in a typical Anaconda distribution of Python:
- OpenCV
- Flask
- Flask Dropzone  
- Flask Uploads
- Keras
- Tensorflow

To install these dependencies, you can follow this procedure:
1. First, set up a virtual environment:
`python -m venv <name of virtual environment`>
2. Activate the virtual environment:
`source <name of virtual environment>/bin/activate`
3. Finally, install the dependencies utilizing the included requirements.txt file:
`pip install -r requirements.txt`

## 2. Project Motivation
This app was submitted as partial fulfillment of the requirements for Udacity's Data Scientist Nanodegree. The project sought to provide hands-on experience with convolutional neural networks (CNN) for image classification tasks. In particular, the objective of the project was to develop an algorithm to firstly detect whether an image contains a human, a dog, or neither. Secondly, the algorithm makes a guess as to the breed of the dog in the image. If an image of a human is submitted, then the algorithm matches the human with the closest looking dog breed. The final deliverable for the project is a Flask app that allows a user to upload an arbitrary image, process the image with the algorithm, and then display the output.

## 3. File Descriptions
The following files are included:
```
dog_app
│
├── README.md
├── LICENSE.txt
├── requirements.txt
├── app.py
│
├── templates
│   ├── index.html
│   ├── uploaded_images.html
│   └── predictions.html
│
└── utility_files
    ├── haarcascade_frontalface_alt.xml
    ├── weights.best.Xception.hdf5
    └── dog_names.pickle
```

- app.py - Contains the main functionality of the app. Notably, the code includes the following elements:
  - Human face recognition algorithm from OpenCV
  - Dog recognition algorithm based on the Resnet50 CNN
  - Dog breed classification model based on transfer learning from the Xception CNN
  - Flask backend for image uploading, classification, and presentation of results
- templates - HTML templates to run the Flask app
- utility_files - Serialized data for loading machine learning algorithms

## 4. Interaction
To run the dog breed classification app, follow the installation instructions above and then run the following from the command line:
1. `export FLASK_APP=app.py`
2. `flask run`

The app can be accessed from a browser at http://127.0.0.1:5000/

## 5. Licensing, Authors, Acknowledgements

- **Licensing**: [MIT License](https://choosealicense.com/licenses/mit/); see LICENSE.txt  
- **Author**: Joshua Tice ([LinkedIn](www.linkedin.com/in/joshuatice))  
- **Acknowledgements**: Many thanks to Dustin D'Avignon, whose blog post and code provided a great head-start for the Flask backend of the app. You can find his blog post and GitHub profile below:  
  - [Medium blog post](https://medium.com/@dustindavignon/upload-multiple-images-with-python-flask-and-flask-dropzone-d5b821829b1d)  
  - [Github profile](https://github.com/ddavignon/flask-multiple-file-upload)
