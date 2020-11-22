Disaster_Response_Pipeline
Table of Contents
Installation
Project Motivation
File Description
Insturctions
Licensing, Authors, and Acknowledgements
Installation
To run the codes the following libraries need to be installed:

Pandas
Numpy
Pickle
Sci-kit Learn
SQL Alchemy
Flask
NLTK
Project Motivation
As part of my Udacity Nanaodegree a web app was created to classify tweets during a disaster.

File Descriptions
data:-process_data.pyETL Pipeline cleaning and formating the data to sql database -disaster_catergories.csvprovided data from Figure-8 -disaster_messages.csv `provided data from Figure-8

models -train_classifier.py ML Pipline training a model and providing a classfier

app -run.py The file used to launch a Flask web app by unsing the provided classifier.

Instructions
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/

Licensing, Authors, Acknowledgements
Must give credit to Figure-8 and Udacity for the data.
