import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Improved dataset to help the model learn differences
data = {
    'Age': [22, 35, 45, 30, 50, 28, 60, 20, 40, 55],
    'Tenure': [6, 24, 60, 12, 72, 8, 48, 2, 36, 65],
    'Monthly_Charges': [500, 1200, 1500, 800, 2000, 650, 100, 900, 400, 300],
    'Contract_Type': ['Month-to-Month', 'One Year', 'Two Year', 'Month-to-Month', 'Two Year', 'One Year', 'Two Year', 'Month-to-Month', 'One Year', 'Two Year'],
    'Internet_Service': ['Fiber', 'DSL', 'Fiber', 'DSL', 'Fiber', 'DSL', 'DSL', 'Fiber', 'DSL', 'Fiber'],
    'Churn': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0] # 1 = Churn, 0 = Stay
}
df = pd.DataFrame(data)

le_contract = LabelEncoder()
le_internet = LabelEncoder()
df['Contract_Type'] = le_contract.fit_transform(df['Contract_Type'])
df['Internet_Service'] = le_internet.fit_transform(df['Internet_Service'])

X = df[['Age', 'Tenure', 'Monthly_Charges', 'Contract_Type', 'Internet_Service']]
y = df['Churn']

model = LogisticRegression()
model.fit(X, y)

with open('churn_model.pkl', 'wb') as f:
    pickle.dump({"model": model, "le_contract": le_contract, "le_internet": le_internet}, f)