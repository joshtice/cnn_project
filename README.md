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

## 2. Project Description  
### Motivation
This app was submitted as partial fulfillment of the requirements for Udacity's Data Scientist Nanodegree. The project sought to provide hands-on experience with convolutional neural networks (CNN) for image classification tasks. 
### Summary  
This project presented multiple approaches to image classification within a production environment. The project began with an introduction to a face detection algorithm from OpenCV to contrast with deep learning methods. Then, a CNN was built from scratch using the Keras framework. Finally, a production-worthy CNN was developed using transfer learning techniques.  

The final deliverable was an app that implements an algorithm to firstly detect whether an image contains a human, a dog, or neither. Then, the algorithm makes a guess as to the breed of the dog in the image. If an image of a human is submitted, then the algorithm matches the human with the closest looking dog breed. The app allows a user to upload an arbitrary image, process the image with the algorithm, and then display the output.  
### Conclusion  
One of the more difficult aspects of the project was setting up the environment to support the training of the CNNs. To implement transfer learning techniques, the Keras framework contains methods for extracting 'bottleneck' features, or representations of input images after they have passed through the convolutional layers of a particular model. Space limitations on the provided cloud platform could not support the files needed to run two CNNs at once, so the training had to be set up on a local machine. Additionally, context managers needed to be set up to allow Tensorflow to run two models within the same app.  

Multiple aspects of the app could be improved:  
- Currently, the face detection algorithm favors front-facing human faces. Faces with a different presentation are more difficult to detect. Another CNN could be used here, such as [Facenet](https://arxiv.org/abs/1503.03832).  
- The dog breed classifier has a propensity for classifying humans as Dachshunds. The fully-connected portion of the neural network is fairly simple, so the architecture could be modified to capture more complexity in the bottleneck features.
- The app currently allows only one image to be uploaded at a time. The infrastructure could be modified to allow multiple files at once.  
- After the user prompts the app to classify the input image, the CNN takes a relatively long time to produce an answer, typically tens of seconds up to over a minute. Perhaps certain techniques, such as [network pruning](https://arxiv.org/abs/1611.06440), could be used to simplify the prediction algorithm to speed up performance.  
- Finally, the user interface for the app is very simple ascetically. The front-end could use a more inviting layout.

Overall, this project was an excellent introduction to pragmatically incorporating deep learning models into production, and it provided many avenues for continued learning.

## 3. File Descriptions
The following files are included:
```
dog_breed_classification_app
│
├── README.md
├── LICENSE.txt
├── requirements.txt
├── .gitignore
├── tice_dog_app_notebook.html
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

Some of the more critical files and directories are explained below:  
- tice_dog_app_notebook.html - The completed jupyter notebook that contains the preliminary analysis for the app
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

The app can then be accessed from a browser at http://127.0.0.1:5000/

## 5. Liensing, Authors, Acknowledgements

- **Licensing**: [MIT License](https://choosealicense.com/licenses/mit/); see LICENSE.txt  
- **Author**: Joshua Tice ([LinkedIn](www.linkedin.com/in/joshuatice))  
- **Acknowledgements**: Many thanks to Dustin D'Avignon, whose blog post and code provided a great head-start for the Flask backend of the app. You can find his blog post and GitHub repository here:  
  - [Medium blog post](https://medium.com/@dustindavignon/upload-multiple-images-with-python-flask-and-flask-dropzone-d5b821829b1d)  
  - [Github repository](https://github.com/ddavignon/flask-multiple-file-upload)
