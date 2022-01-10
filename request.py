import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Senior Citizen':'Yes', 
                            'Partner':'No', 
                            'Dependents':'Yes',
                            'Tenure':13,
                            'Contract':'Month-to-month',
                             'Paperless Billing':'No',
                             'Phone service':'No',
                             'Multiple Lines':'No',
                             'Internet Service_Fiber optic': 'Yes',
                             'Online Security': 'Yes',
                             'Online Backup':'No',
                             'Device Protection':'Yes',
                             'Tech Support':'No',
                             'Streaming TV':'Yes',
                             'Internet service':'Yes',
                              'Streaming Movies':'No',
                              'Payment Method_Credit card':'Yes',
                              'Payment Method_Electronic check':'No',
                              'Payment Method_Mailed check':'No',
                              'Gender':'Male'
                            })

print(r.json())