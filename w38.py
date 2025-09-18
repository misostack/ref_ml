"""TODO
source .venv/bin/activate
1. install pandas
uv add pandas
2. Data Exploration
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime
import math
import pandas as pd

melb_data = pd.read_csv('./w38/melb_data.csv', sep=',')
rows, columns = melb_data.shape
print(f'Rows: {rows}, Columns: {columns}')
# print(melb_data.info())
# print(melb_data.describe())
# print(melb_data.head())

# home_data = melb_data.describe()


# for k, v in home_data.items():
#     print(f'{k}: Count: {v.count()}')

# avg_lot_size = math.ceil(home_data["LotArea"]["mean"])

# # As of today, how old is the newest home (current year - the date in which it was built)
# newest_home_age = datetime.now().year - \
#     math.ceil(home_data["YearBuilt"]["mean"])

# # Checks your answers
# print(f'Average Lot Size: {avg_lot_size}')
# print(f'Newest Home Age: {newest_home_age}')

print(melb_data.columns)

# drop missing values
melb_data = melb_data.dropna(axis=0)

# prediction target is price
y = melb_data.Price

# choosing some features
melb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# By convention, this data is called X.
X = melb_data[melb_features]

print(X.describe())

print(X.head())

# training

# define model : specify the type of model you want to use
melb_model = DecisionTreeRegressor(random_state=1)

# Fit model : capture patterns in data
# X: features, y: target
melb_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
predict_with_head = melb_model.predict(X.head(100))
print(mean_absolute_error(y.head(100), predict_with_head))

# compare predictions to actual prices
print("The actual prices are")
print(y.head())

predicted_home_prices = melb_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))
# mea: Sai số tuyệt đối trung bình - mean absolute error
# https://www.datacamp.com/tutorial/loss-function-in-machine-learning


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

# Underfitting and Overfitting


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return (mae)


for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %
          (max_leaf_nodes, my_mae))

[5, 50, 500, 5000][0:0]

scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y)
          for leaf_size in [5, 50, 500, 5000]}
best_tree_size = min(scores, key=scores.get)
print(f'Best tree size: {best_tree_size}')

# Random Forest


forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)
dummy_mean = DummyRegressor(strategy="mean")
dummy_mean.fit(X_train, y_train)
y_pred = dummy_mean.predict(X_val)

print("Baseline predictions:", y_pred)
print("Baseline MAE:", mean_absolute_error(y_val, y_pred))

dummy_rmse = DummyRegressor(strategy="median")
dummy_rmse.fit(X_train, y_train)
y_pred = dummy_rmse.predict(X_val)

print("Baseline predictions:", y_pred)
print("Baseline RMSE:", root_mean_squared_error(y_val, y_pred))
