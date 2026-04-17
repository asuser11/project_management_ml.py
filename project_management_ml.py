import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Sample dataset
data = {
    'Duration': [30, 45, 60, 20, 90],
    'Budget': [100000, 150000, 200000, 80000, 300000],
    'TeamSize': [5, 8, 10, 4, 12],
    'Progress': [80, 60, 50, 90, 30],
    'Delay': [0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['Duration', 'Budget', 'TeamSize', 'Progress']]
y = df['Delay']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

prediction = model.predict([[50, 120000, 6, 70]])

print("Will project be delayed? (1=Yes, 0=No):", prediction[0])