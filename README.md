# Customer Churn Prediction
## *Building a Model to Detect Churn in a Telco Company*
## ________________________________________________________

### Project Structure
1. Build a machine learning model
  - Find 'customer_churn_ML_notebook.ipynb' and run all lines to use the notebook.
2. Create an API for the final model using FLASK

### How to Use the API
1. GO to home directory of the project and the command:
```
python model.py
```
Running this command line will create a serialized model 'model.pkl'

2. Run api.py to initialize FLASK with this command:
```
python api.py
```
The model should be loaded, and you should be able to see an URL http://127.0.0.1:12345/ as shown in the picture "initialize flask.png".

3. Once you see the homepage, please give input to each box. See the picture "homepage".
Input guidelines:
  - Tenure: any number between 0 and 72.
  - Contract: a string of one of these (Month-to-month, One year, Two year).
  - Gender: a string of these (Male, Female),
  - Other variables should have an input of either Yes or No.
Click on 'Predict' to generate the result. See the picture 'prediction result'.
