## ML-Model-Flask-Deployment
This is a demo project to elaborate how Machine Learn Models are deployed on production using Flask API

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. iris_model.py - This contains code fot our Machine Learning model to predict iris classification absed on trainign data in 'iris.csv' file.
2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.
3. forms.py - This uses flask forms to create and validate input parameters
4. templates - This folder contains the HTML template to allow user to enter employee detail and displays the predicted employee salary.

