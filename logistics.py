import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("escape_velocity.csv")

velocity_list = df["Velocity"].tolist()
escaped_list = df["Escaped"].tolist()

fig = px.scatter(x=velocity_list, y=escaped_list)
fig.show()

import numpy as np
velocity_array = np.array(velocity_list)
escaped_array = np.array(escaped_list)

m, c = np.polyfit(velocity_array, escaped_array, 1)

y = []
for x in velocity_array:
  y_value = m*x + c
  y.append(y_value)

fig = px.scatter(x=velocity_array, y=escaped_array)
fig.update_layout(shapes=[
    dict(
      type= 'line',
      y0= min(y), y1= max(y),
      x0= min(velocity_array), x1= max(velocity_array)
    )
])
fig.show()

X=np.reshape(velocity_list,(len(velocity_list),1))
Y=np.reshape(escaped_list,(len(escaped_list),1))

lr = LogisticRegression()
lr.fit(X, Y)

plt.figure()
plt.scatter(X.ravel(),Y,color='black',zorder=20)

def model(x):
  return 1 / (1 + np.exp(-x))

X_test = np.linspace(0, 100, 200)
chances = model(X_test * lr.coef_ + lr.intercept_).ravel()

plt.plot(X_test, chances, color='red', linewidth=3)
plt.axhline(y=0, color='k', linestyle='-')
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=0.5, color='b', linestyle='--')

plt.axvline(x=X_test[165], color='b', linestyle='--')

plt.ylabel('y')
plt.xlabel('X')
plt.xlim(75, 85)
plt.show()

user_score = float(input("Enter your velocity here:- "))
chances = model(user_score * lr.coef_ + lr.intercept_).ravel()[0]
if chances <= 0.01:
    print("You will not get accepted.")
elif chances >= 1:
    print("You will get accepted.")
elif chances < 0.5:
    print("You might not gect accepted.")
else:
    print("You may get accepted.")
    
