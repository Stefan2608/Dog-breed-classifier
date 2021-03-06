# Dog breed classifier

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Directories](#directories)
4. [File Description](#files)
5. [Instructions](#instructions)
6. [Summary](#summary)
7. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

You can find all libraries you need in the requirements.txt

## Project Motivation <a name=motivation"></a>
 
This is the final project of my Udacity training to become a data scientist. The entire training was a lot of fun and I learned a lot and had to apply everything I learned in this project. I hope have fun using the dog breed classifier
 
 ## Directories <a name="directories"></a>

   - templates - directory with HTML files used by the web application
   - respository - directory with trained model and neccessary data for the web application 
   - images - folder where the uploaded pictures where saved temporarily
   - requirements.txt - all python libraries

## File Descriptions <a name="files"></a>

   - Web application:  -dog_breed_application.py`launch a Flask web app to predict dog breed in pictures.
   - Juphyter notebook:-dog_app.ipynb`Workbook of the project, if you have any special interest in more information.
   - Project Overview: -project_overview.pdf`This file provides an Overview how the project was approached.
          
  
## Instructions<a name="instructions"></a>


1.Take care that the link for the image directory in the dog_breed_application is correct
"IMAGE_FOLDER = './images/' # image filepath" 

2. Run the following command in the app's directory to run your web app.
    `python dog_breed_applicaiton.py`

3. Go to http://0.0.0.0:3001/

4. Upload a picture and hit the button "Upload image". Your picture will now checked by the algorithm if its rather a dog or human and predicting the most fitting breed. 

This is how the Applicaiton should look like! 

![Webapp](https://github.com/Stefan2608/Dog-breed-classifier/blob/main/WebApplication.png?raw=true)


## Summary 

This web application is based on three functions:

1. Dog Detector: The uploaded picture checks whether there is a dog on it.
2. Human Face Detector: the uploaded image is checked for human faces.
3. Dog breed classifier: This CNN, pre-trained with the help of a ResNet50, checks the images in which a dog or a human face was recognized and assigns them the most similar    breed of dog with an accuracy of 84,6 %.

Result is one of three messages: 
![Lab](https://github.com/Stefan2608/Dog-breed-classifier/blob/main/images/Lab.png?raw=true)
![car](https://github.com/Stefan2608/Dog-breed-classifier/blob/main/images/car.png?raw=true)
![human](https://github.com/Stefan2608/Dog-breed-classifier/blob/main/images/human.png?raw=true)

The exact procedure is explained in Project Overview. Feel free to check it out!

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Thank you Udacity for providing the project data and this training
