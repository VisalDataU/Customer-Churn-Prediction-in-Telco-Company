import pandas as pd
import numpy as np


data =  pd.read_excel('Merged_Data.xlsx')

# Drop columns from data
colsToDrop = ['Tenure Months', 'Count', 'Churn Label', 
              'LoyaltyID', 'Zip Code', 'State', 'Lat Long',
              'Latitude', 'Longitude', 'Quarter', 'Country', 'Churn Category',
              'City', 'Customer Status', 'Customer ID', 'CustomerID', 
              'Churn Reason', 'Churn Score', 'Churn Value',
              'CLTV', 'Satisfaction Score', 'Churn Category']
data.drop(colsToDrop, axis=1, inplace=True)

#Fill null values in 'Churn Category' and 'Churn Reason' columns
data['Churn Reason'] = data['Churn Reason'].fillna('Not Churned')

#Convert 'Contract' column to category type
data['Contract'] = data['Contract'].astype('category')
data['Payment Method'] = data['Payment Method'].astype('category')
data['Gender'] = data['Gender'].astype('category')
data['Churn Reason'] = data['Churn Reason'].astype('category')


# Set order of categories to 'Contract'
data['Contract'] = data['Contract'].cat.set_categories(
                                new_categories=['Month-to-month', 'One year', 'Two year'],
                                ordered=True)


#Convert str to flaot
data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors='coerce')


#Drop rows that have missing values in 'Total Charges' column because the missing values account only 0.15% of the data
data.dropna(inplace=True)

# Reset index
data.reset_index(drop=True, inplace=True)


# Create a new 'Churn Category'
values = ['Attitude', 'Competitor', 'Dissatisfaction', 'Price', 'Not Churned']
conditions = list(map(data['Churn Reason'].str.contains, values))

data['Churn Category'] = np.select(conditions, values, 'Other')

#Defining the map function
def binary_map(feature):
    return feature.map({'Yes':1, 'No':0})

## Encoding target feature
data['Churn'] = data[['Churn']].apply(binary_map)

#Encoding other binary category
binary_list = ['Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Paperless Billing']
data[binary_list] = data[binary_list].apply(binary_map)

# Encoding gender category
data['Gender'] = data['Gender'].map({'Male':1, 'Female':0})

# Encoding orninal variable
mapper = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
data['Contract'] = data['Contract'].replace(mapper)

# Encoding the other categoric features with more than two categories
data = pd.get_dummies(data, drop_first=True)



#Drop columns which are correlated
colsToDrop = ['Internet Service_No', 'Online Security_No internet service',
              'Online Backup_No internet service', 'Device Protection_No internet service', 'Tech Support_No internet service',
              'Streaming TV_No internet service', 'Phone Service', 'Total Charges', 'Monthly Charges']

data.drop(colsToDrop, axis=1, inplace=True)


# select independent variables
X = data.drop(columns='Churn')

# select dependent variables
y = data.loc[:, 'Churn']


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Create parameter grid.
param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]

# Make a pipeline to include `StandardScaler()` and `GridSearchCV()`
pipe_LR = make_pipeline(StandardScaler(),
                        GridSearchCV(LogisticRegression(random_state=1),
                                     param_grid=param_grid,
                                     cv = 5,
                                     verbose=True,
                                     n_jobs=-1,
                                     refit=True)
)

# Fit pipeline
gridresult = pipe_LR.fit(X_train, y_train)

# Best estimator
gridresult.named_steps['gridsearchcv'].best_estimator_

# Best params
gridresult.named_steps['gridsearchcv'].best_params_

# Save model
import pickle
pickle.dump(gridresult, open('model.pkl','wb'))
print("Model dumped!")

# Load the model back
lr = pickle.load(open('model.pkl','rb'))