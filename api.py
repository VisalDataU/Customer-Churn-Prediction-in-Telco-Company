# Dependencies
from flask import Flask, request, jsonify, render_template
import pickle
import traceback
import sys
import pandas as pd
import numpy as np
import json

# Your API definition
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    #if lr:
        #try:
    json_ = request.form.to_dict()
    #print(json_)
    data_1 = pd.DataFrame([json_])
    def binary_map(feature):
        return feature.map({'Yes':1, 'No':0})
    binary_list = ['Senior Citizen', 'Partner', 'Dependents', 'Paperless Billing', 'Multiple Lines_No phone service', 
                    'Multiple Lines_Yes', 'Internet Service_Fiber optic', 'Online Backup_Yes', 'Online Security_Yes', 
                    'Device Protection_Yes', 'Tech Support_Yes', 'Streaming TV_Yes', 'Streaming Movies_No internet service', 'Streaming Movies_Yes', 
                    'Payment Method_Electronic check', 'Payment Method_Mailed check', 'Payment Method_Credit card (automatic)']
    data_1[binary_list] = data_1[binary_list].apply(binary_map)
    mapper = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
    data_1['Contract'] = data_1['Contract'].replace(mapper)
    # Encoding gender category
    data_1['Gender_1'] = data_1['Gender_1'].map({'Male':1, 'Female':0})
    data_1['Tenure'] = data_1['Tenure'].astype(float)
    
    result = lr.predict(data_1)
    if result == [0]:
        prediction = 'not churn'
    else: prediction = 'churn'

    return render_template('index.html', prediction_text='The customer will {}.'.format(prediction))

    #     except:
            
    #         return jsonify({'trace': traceback.format_exc()})
    # else:
        
    #     print ('Train the model first')
        
    #     return ('No model here to use')


    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls throught request
    '''
    data = request.get_json(force=True)
    #query_1 = pd.get_dummies(pd.DataFrame(data))
    data_2 = pd.DataFrame([data])
    #query = pd.get_dummies(pd.DataFrame([json_]))
    def binary_map(feature):
        return feature.map({'Yes':1, 'No':0})
    binary_list = ['Senior Citizen', 'Partner', 'Dependents', 'Paperless Billing', 'Multiple Lines_No phone service', 
                   'Multiple Lines_Yes', 'Internet Service_Fiber optic', 'Online Backup_Yes', 'Online Security_Yes',
                   'Device Protection_Yes', 'Tech Support_Yes', 'Streaming TV_Yes', 
                   'Streaming Movies_No internet service', 'Streaming Movies_Yes', 
                   'Payment Method_Electronic check', 'Payment Method_Mailed check', 'Payment Method_Credit card (automatic)']
    data_2[binary_list] = data_2[binary_list].apply(binary_map)
    mapper = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
    data_2['Contract'] = data_2['Contract'].replace(mapper)
    # Encoding gender category
    data_2['Gender_1'] = data_2['Gender_1'].map({'Male':1, 'Female':0})
    data_2['Tenure'] = data_2['Tenure'].astype(float)
    prediction_1 = list(lr.predict(data_2))

    output = prediction_1[0]
    return jsonify(output)

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 12345 # If you don't provide any port then the port will be set to 12345
    lr = pickle.load(open('model.pkl','rb')) # Load "model.pkl"
    print ('Model loaded')
    model_columns = pickle.load(open('model_columns.pkl', 'rb')) # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(port=port, debug=True)
    


