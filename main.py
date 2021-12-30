import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

file = "Chicago_hotels.csv"
x_label = "Month"
y_label = "Hotel Occupancy % according to Chicago Central Business District Hotel Statistics"
test_size = 0.0707
random_state = 0

df = pd.read_csv(file, header=0, encoding='windows-1251', delimiter=';', decimal=',', escapechar=' ')
col = ['x1']
pd.options.mode.chained_assignment = None

X = df[col].index.values.reshape(-1, 1)
y = df[col].values.reshape(-1, 1)

plt.plot(X, y)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

regression = LinearRegression()
regression.fit(X_train, y_train)
y_predicted = regression.predict(X)
y_test_predicted = regression.predict(X_test)

plt.figure(figsize=(10, 8))
plt.scatter(X, y)
plt.plot(X, y_predicted, color='red')
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.tight_layout()
plt.show()

df0 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_test_predicted.flatten()})
df1 = df0.head(25)
df1.plot(kind='bar')
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle='-')
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.tight_layout()
plt.show()

plt.scatter(X_test, y_test)
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test_predicted)
plt.plot(X_test, y_test_predicted, color='red')
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.tight_layout()
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_predicted))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_predicted))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predicted)))

print('Predicted values:')
print(y_test_predicted.flatten())
