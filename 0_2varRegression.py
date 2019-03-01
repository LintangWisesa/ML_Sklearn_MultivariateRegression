# multivariate linear regression
# linear regression with multiple variables
# y = m1x1 + m2x2 + b

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataRumah = {
    'luas': [2600, 3000, 3200, 3600, 4000],
    'kamar': [3, 4, 1, 3, 5],
    'harga': [550000, 565000, 610000, 595000, 760000]
}
df = pd.DataFrame(dataRumah)
print(df)

# ==============================

from sklearn import linear_model
model = linear_model.LinearRegression()

# training
model.fit(df[['luas', 'kamar']], df['harga'])

# slope m1 & m2
print(model.coef_)

# intercept b
print(model.intercept_)

# ==============================

# prediction utk luas: 3200, kamar: 2
print(model.predict([[3600, 3]]))