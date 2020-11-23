# Dog breed classifier

### Table of Contents

1. [Installation](#installation)
2. [Directories](#directories)
3. [File Description](#files)
4. [Insturctions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
 To run the codes the following libraries need to be installed:
 
- Pandas
- Keras
- os
- tqdm
- cv2
- Numpy
- Pickle
- Sci-kit Learn
- SQL Alchemy
- Flask

## Directories <a name="directories"></a>

   - templates - directory with HTML files used by the web application
   - respository - directory with trained model and neccessary data for the web application 
   - images - folder where the uploaded pictures where saved temporarily

## File Descriptions <a name="files"></a>

 
   * `Web application:`  -dog_breed_application.py`launch a Flask web app to predict dog breed in pictures 
   * `Juphyter notebook:`-dog_app.ipynb`Workbook of the project, if you have any special interest in more information
          
  
## Instructions<a name="instructions"></a>

1. Take care that the link for the image directory in the dog_breed_application is correct 
"IMAGE_FOLDER = './images/' # image filepath" 

2. Run the following command in the app's directory to run your web app.
    `python dog_breed_applicaiton.py`

3. Go to http://0.0.0.0:3001/


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Thank you Udacity for providing the project data! 
