from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('dataset.csv')
df = df.dropna()
# ------------------------------------------------------------------------------------------------------------------


df = df.drop(columns=['Formatted Date'])

# ------------------------------------------------------------------------------------------------------------------

encoder = LabelEncoder()
# df['Partly Cloudy'] = encoder.fit_transform(df['Partly Cloudy'])
df['Daily Summary'] = encoder.fit_transform(df['Daily Summary'])
df['Summary'] = encoder.fit_transform(df['Summary'])
df['Precip Type'] = encoder.fit_transform(df['Precip Type'])

x = df.drop(columns=['Daily Summary'])

y = df['Daily Summary']

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

model = RandomForestRegressor()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

# accuracy = accuracy_score(y_test, prediction)
# Evaluate the regression model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# print(
#     f"output of the linear regression model for weather prediction is: {prediction}"
# )
