#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:22:59 2024

@author: meghanapuli
"""

'''
Problem statement

Suppose that you are the administrator of a university department and you want to determine each applicant’s chance of admission based on their results on two exams. 
* You have historical data from previous applicants that you can use as a training set for logistic regression. 
* For each training example, you have the applicant’s scores on two exams and the admissions decision. 
* Your task is to build a classification model that estimates an applicant’s probability of admission based on the scores from those two exams.

y_train = 1 if the student was admitted 
y_train = 0 if the student was not admitted
'''

import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data = np.loadtxt("test_scores.txt", delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y

# Load the dataset
X_train, y_train = load_data()

def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0
    
    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)
    
# Plot training data
print("\nTraining data")
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()

def sigmoid(z):
      
    g = 1.0/(1.0+np.exp(-z))

    return g

# Compute the prediction of the model
def compute_model_output(X, w, b): 
    z = np.dot(w,X) + b
    f_wb = sigmoid(z)
    return f_wb

# Predict function to produce 0 or 1 predictions given a dataset
def predict(X, w, b): 

    m, n = X.shape   
    p = np.zeros(m)

    # Loop over each example
    for i in range(m):   
        z_wb = 0
        # Loop over each feature
        for j in range(n): 
            # Add the corresponding term to z_wb
            z_wb += X[i,j] * w[j]
        
        # Add bias term 
        z_wb += b
        
        # Calculate the prediction for this example
        f_wb = sigmoid(z_wb)

        # Apply the threshold
        if f_wb >= 0.5:
            p[i] = 1
        else:
            p[i] = 0

    return p

# Compute the cost of the model
def compute_cost(X, y, w, b):

    m, n = X.shape

    total_cost = 0
    for i in range(m):
        z_i = np.dot(w,X[i]) + b
        f_wb_i = sigmoid(z_i)
        total_cost += -y[i]*np.log(f_wb_i) - (1 - y[i])*np.log(1 - f_wb_i)
    total_cost = total_cost / m
    
    return total_cost

# Compute the gradient
def compute_gradient(X, y, w, b): 

    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_wb = np.dot(w,X[i]) + b
        f_wb = sigmoid(z_wb)
        err_i = f_wb - y[i]
        
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]
        
        dj_db += err_i
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

# Gradient descent to find optimal w,b
def gradient_descent(X, y, w_in, b_in, gradient_function, alpha, num_iters): 

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db   
        
    return w_in, b_in

w_tmp  = 0.01 * (np.random.rand(2) - 0.5)
b_tmp  = -8

alpha = 0.001
num_iters = 10000

w_out, b_out = gradient_descent(X_train, y_train, w_tmp, b_tmp, compute_gradient, alpha, num_iters)            
 
print(f"\nOptimal parameters: w:{w_out}, b:{b_out}")

cost = compute_cost(X_train, y_train, w_out, b_out)
print("Cost of our model: ",cost)

# Functions to plot the decision boundary
def map_feature(X1, X2):

    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)

def plot_decision_boundary(w, b, X, y):

    plot_data(X[:, 0:2], y)
    
    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)
        
        plt.plot(plot_x, plot_y, c="b")
        
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        
        z = np.zeros((len(u), len(v)))

        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = sigmoid(np.dot(map_feature(u[i], v[j]), w) + b)
     
        z = z.T

        plt.contour(u,v,z, levels = [0.5], colors="g")

print("\nDecision boundary")
plot_decision_boundary(w_out, b_out, X_train, y_train)
# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()

# Compute accuracy on our training set
p = predict(X_train, w_out, b_out)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))

print("\nPlease enter the scores of the applicant")
X_test = []
X_test.append(float(input("\nExam 1 score: ")))
X_test.append(float(input("Exam 2 score: ")))

computed_value = compute_model_output(X_test, w_out, b_out)
print("\nApplicant’s probability of admission: ",computed_value)

if computed_value < 0.5:
    output = 0
    print("Result: Not Admitted")
    
else:
    output = 1
    print("Result: Admitted")