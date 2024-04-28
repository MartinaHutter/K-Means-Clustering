!pip install rfpimp

import rfpimp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# === LOAD DATA ===
dftrain1 = pd.read_excel('test8.xlsx') # load whole dataset
df = dftrain1.iloc[:, :]

# set up features
features = ['Type', 'A', 'B', 'Actual B', 'Time', 'Theta', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10','P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20']

# split training and evaluation set
df_train, df_test = train_test_split(dftrain1, test_size=0.20)
df_train = df_train[features]
df_test = df_test[features]

# drop parameter P2 as a response variable
X_train, y_train = df_train.drop('P2',axis=1), df_train['P2']
X_test, y_test = df_test.drop('P2',axis=1), df_test['P2']

# ==================



# ===GENERATE CORRELATION MATRIX ===
corr = df.corr(method='spearman')

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(10, 9)) #6 5

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)

fig.suptitle('Correlation matrix of features', fontsize=15)


fig.tight_layout()

# ===========END OF CORRELATION MATRIX=================


# Regression trining
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)

# importances of features
imp = rfpimp.importances(rf, X_test, y_test)
print(imp)

# Plot Importances
plt.figure(figsize=(10, 8))
imp.plot(kind='barh', color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important features at the top
plt.show()

#======================

# Regression for a feature with highest iportance - P14
X = dftrain1['P14'].values.reshape(-1,1)
y = dftrain1['P2'].values

#Training
ols = linear_model.LinearRegression()
model = ols.fit(X, y)
response = model.predict(X)

#Eval
r2 = model.score(X, y)

#2D plot of regression
plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(X, response, color='r', label='Regression model')
ax.scatter(X, y, edgecolor='r', facecolor='blue', alpha=0.7, label='Sample data')
ax.set_ylabel('P2', fontsize=14)
ax.set_xlabel('P14', fontsize=14)

ax.legend(facecolor='white', fontsize=11)
ax.set_title('$R^2= %.2f$' % r2, fontsize=18)

fig.tight_layout()

#==========================


# Try to predict P2 for a random sample

features = ['Type', 'A', 'B', 'Actual B', 'Time', 'Theta', 'P1', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10','P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20']
target = 'P2'

X = dftrain1[features].values.reshape(-1, len(features))
y = dftrain1[target].values

ols = linear_model.LinearRegression()
model = ols.fit(X, y)

model.score(X, y)

# Insert the datapoint values
x_pred = np.array([1, 2000, 96.61, 96.66, 61.09, 87.06, 0.3, 80, -1000, 1000, 10, 3, 15, 5, 40000, -50000, 50000, 20, 18, 16, 14, 12, 11, 10, 9])
x_pred = x_pred.reshape(-1, len(features))

# RESULT
model.predict(x_pred)

