import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('linear_data.csv')

# plt.scatter(data.YearsExperience, data.Salary)
# plt.show()


def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        print(x)
        print(y)
        total_error += (y-(m*x+b))**2
    total_error / float(len(points))


def gradient_descent(m_now, b_now, points, lr):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary

        # print(x)
        # print(y)

        # dm = -(2/n)*x*(y-y_predict)
        m_gradient += -(2/n) * x * (y - (m_now*x + b_now))
        # db =  -(2/n)*(y-y_predict)
        b_gradient += -(2/n) * (y - (m_now*x + b_now))

    m = m_now - lr * m_gradient
    b = b_now - lr * b_gradient

    return m, b


m = 0
b = 0
lr = 0.001
n_iterations = 10000

for i in range(n_iterations):
    m, b = gradient_descent(m, b, data, lr)

print(m, b)


plt.scatter(data.YearsExperience, data.Salary)
plt.plot(list(range(1, 11)), [m*x + b for x in range(1, 11)], color="black")
plt.show()
