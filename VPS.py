import pandas as pd

# Load the dataset
car_data = pd.read_csv("car_data.csv")
# Basic overview
print(car_data.shape)
print(car_data.head())

# Drop columns with too many missing values
car_data = car_data.drop(['Invoice Price','Cylinders','Highway Fuel Economy'], axis=1)

# Extract numeric values
car_data['Horsepower_no'] = car_data['Horsepower'].str.extract('(\d+)').astype(float)
car_data['Torque_no'] = car_data['Torque'].str.extract('(\d+)').astype(float)

car_data['MSRP'] = car_data['MSRP'].str.replace('[$,]', '', regex=True).astype(float)
car_data['Used/New Price'] = car_data['Used/New Price'].str.replace('[$,]', '', regex=True).astype(float)

import seaborn as sns
import numpy as np
sns.pairplot(car_data[['MSRP', 'Horsepower_no', 'Torque_no']])

import matplotlib.pyplot as plt
sns.barplot(x='Body Size', y='MSRP', data=car_data)
plt.xticks(rotation=45)
plt.show()

# Step 4: 🧠 Train the Linear Regression Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

car_data = car_data.dropna()                  # Drop NaNs
#car_data = car_data[~np.isinf(car_data).any(axis=1)] # Drop rows with infinities

X = car_data[['Horsepower_no', 'Torque_no']]
y = car_data['MSRP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lm = LinearRegression()
lm.fit(X_train, y_train)

horsepower = 150
torque = 300
features = np.array([[horsepower, torque]])
price = lm.predict(features)
print(f"Estimated Price: ${price[0]:,.2f}")

#Step 5: 💾 Save the Model
import pickle
with open("linear_model.pkl", "wb") as f:
    pickle.dump(lm, f)


