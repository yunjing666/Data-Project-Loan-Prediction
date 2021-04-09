# Import required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Read the dataset
loan = pd.read_csv('./data/loan.csv')
print(loan.head())
print(loan.info())

print('\n\nColumn Names\n\n')
print(loan.columns)

#Label encode the target variable
encode = LabelEncoder()
loan.Loan_Status = encode.fit_transform(loan.Loan_Status)

# Drop the null values
loan.dropna(how='any',inplace=True)


# Train and test data
train, test = train_test_split(loan,test_size=0.2,random_state=0)



# Seperate the target and independent variable
train_x = train.drop(columns=['Loan_ID','Loan_Status'],axis=1)
train_y = train['Loan_Status']
print(train_x.head())
print(train_y.head())

test_x = test.drop(columns=['Loan_ID','Loan_Status'],axis=1)
test_y = test['Loan_Status']

# Encode the data
train_x = pd.get_dummies(train_x)
test_x  = pd.get_dummies(test_x)

print('Shape of training data : ',train_x.shape)
print('Shape of testing data : ',test_x.shape)

# Fit and evaluate the model
model = LogisticRegression()

model.fit(train_x,train_y)

predict = model.predict(test_x)

print('Predicted Values on Test Data',predict)

print('\n\nAccuracy Score on test data : \n\n')
print(accuracy_score(test_y,predict))