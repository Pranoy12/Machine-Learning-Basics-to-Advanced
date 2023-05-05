#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('salary.csv')
print(data)


plt.scatter(data.YearsExperience, data.Salary)


def loss_function(m, c, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        total_error += (y-(m*x-c))**2
    total_error/float(len(points))


def gradient_descent(m_now, c_now, points, L):
    m_gradient = 0
    c_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary

        m_gradient += -(2/n)*x*(y-(m_now*x+c_now))
        c_gradient += -(2/n)*(y-(m_now*x+c_now))

    m = m_now-L*m_gradient
    c = c_now-L*c_gradient

    return m, c


m = 0
c = 0
L = 0.0001
epochs = 1000

for i in range(epochs):
    m, c = gradient_descent(m, c, data, L)

print(m, c)

plt.scatter(data.YearsExperience, data.Salary, color="black")
plt.plot(list(range(2, 10)), [m*x+c for x in range(2, 10)], color="red")
plt.show
