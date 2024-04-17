# This is a Python script which calculates the logistic regression for the
# ChallengerData data set.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression


# The Data

x = np.array([53,57,58,63,66,67,67,67,68,69,70,70,70,70,72,73,75,75,76,76,78,79,81]).reshape(-1, 1)
y = np.array([1,  1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]).reshape(-1, 1)
# logistic regression model
logreg = LogisticRegression().fit(x, y)

# print coefficients
print(logreg.intercept_, logreg.coef_[0])
#[0.52055518] [-0.02100215]


import seaborn as sns; sns.set_theme(color_codes=True)
df = pd.DataFrame( {'Temperature': x.flatten(), 'Failure': y.flatten()})

ax = plt.gca()
ax.set(xlim = [31,82])
sns.regplot(x='Temperature', y='Failure',  data = df, logistic=True, ax=ax)
plt.title('Challenger O-Ring data')
plt.show()
#  plt.savefig("Oring_logistic_regression.pdf")

ax = plt.gca()
ax.set(xlim = [31,82])
sns.scatterplot(x='Temperature', y='Failure',  data = df, ax=ax)
plt.title('Challenger O-Ring data')
plt.show()

# plt.savefig("O-Ring Data.pdf")