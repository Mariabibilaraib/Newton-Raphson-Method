#!/usr/bin/env python
# coding: utf-8

# Certainly! The Newton-Raphson method is an iterative numerical technique for finding the roots of a real-valued function. Let's consider an example problem and demonstrate the Newton-Raphson method graphically using Python. In this example, we'll solve the equation f(x) = 6x^3 - 6x^2 + 11x - 6
# 

# Here's a Python script using the matplotlib library to graphically demonstrate the Newton-Raphson method:

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Function definition
def f(x):
    return x**3 - 6*x**2 + 11*x - 6

# Derivative of the function
def df(x):
    return 3*x**2 - 12*x + 11

# Newton-Raphson method
def newton_raphson(initial_guess, tolerance=1e-6, max_iterations=100):
    x = initial_guess
    iterations = 0

    while iterations < max_iterations:
        x_new = x - f(x) / df(x)
        if np.abs(x_new - x) < tolerance:
            return x_new, iterations
        x = x_new
        iterations += 1

    raise Exception("Newton-Raphson method did not converge")

# Generate x values for plotting
x_values = np.linspace(0, 4, 1000)

# Plot the function
plt.plot(x_values, f(x_values), label='f(x) = $x^3 - 6x^2 + 11x - 6$')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--', label='y=0')

# Initial guess
initial_guess = 2.0
plt.scatter(initial_guess, f(initial_guess), color='red', marker='o', label='Initial Guess')

# Apply Newton-Raphson method
root, iterations = newton_raphson(initial_guess)
plt.scatter(root, f(root), color='green', marker='o', label=f'Root (after {iterations} iterations)')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Newton-Raphson Method for Finding Root of a Nonlinear Function')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


# This script defines the function, its derivative, and the Newton-Raphson method. It then plots the function, the initial guess, and the final root found by the Newton-Raphson method. The plot helps visualize how the method converges to the root.

# Certainly! Here's the Python script for graphically demonstrating the Newton-Raphson method for the function 
# 
# f(x)=sin(x)−x:

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Function definition
def f(x):
    return np.sin(x) - x

# Derivative of the function
def df(x):
    return np.cos(x) - 1

# Newton-Raphson method
def newton_raphson(initial_guess, tolerance=1e-6, max_iterations=100):
    x = initial_guess
    iterations = 0

    while iterations < max_iterations:
        x_new = x - f(x) / df(x)
        if np.abs(x_new - x) < tolerance:
            return x_new, iterations
        x = x_new
        iterations += 1

    raise Exception("Newton-Raphson method did not converge")

# Generate x values for plotting
x_values = np.linspace(-2, 2, 1000)

# Plot the function
plt.plot(x_values, f(x_values), label='$f(x) = \sin(x) - x$')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--', label='y=0')

# Initial guess
initial_guess = 0.5
plt.scatter(initial_guess, f(initial_guess), color='red', marker='o', label='Initial Guess')

# Apply Newton-Raphson method
root, iterations = newton_raphson(initial_guess)
plt.scatter(root, f(root), color='green', marker='o', label=f'Root (after {iterations} iterations)')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Newton-Raphson Method for Finding Root of $f(x) = \sin(x) - x$')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


# Certainly! Here's a Python script that graphically demonstrates the Newton-Raphson method for the function 
# 
# f(x)=tan(x)−x^3
# 

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

# Function definition
def f(x):
    return np.tan(x) - x**3

# Derivative of the function
def df(x):
    return 1/np.cos(x)**2 - 3*x**2

# Newton-Raphson method
def newton_raphson(initial_guess, tolerance=1e-6, max_iterations=100):
    x = initial_guess
    iterations = 0

    while iterations < max_iterations:
        x_new = x - f(x) / df(x)
        if np.abs(x_new - x) < tolerance:
            return x_new, iterations
        x = x_new
        iterations += 1

    raise Exception("Newton-Raphson method did not converge")

# Generate x values for plotting
x_values = np.linspace(-2, 2, 1000)

# Plot the function
plt.plot(x_values, f(x_values), label='$f(x) = \\tan(x) - x^3$')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--', label='y=0')

# Initial guess
initial_guess = 1.0
plt.scatter(initial_guess, f(initial_guess), color='red', marker='o', label='Initial Guess')

# Apply Newton-Raphson method
root, iterations = newton_raphson(initial_guess)
plt.scatter(root, f(root), color='green', marker='o', label=f'Root (after {iterations} iterations)')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Newton-Raphson Method for Finding Root of $f(x) = \\tan(x) - x^3$')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


#  Here's a Python script that graphically demonstrates the Newton-Raphson method for the function 
# 
# 
# f(x)=x^2−4:

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# Function definition
def f(x):
    return x**2 - 4

# Derivative of the function
def df(x):
    return 2*x

# Newton-Raphson method
def newton_raphson(initial_guess, tolerance=1e-6, max_iterations=100):
    x = initial_guess
    iterations = 0

    while iterations < max_iterations:
        x_new = x - f(x) / df(x)
        if np.abs(x_new - x) < tolerance:
            return x_new, iterations
        x = x_new
        iterations += 1

    raise Exception("Newton-Raphson method did not converge")

# Generate x values for plotting
x_values = np.linspace(-3, 3, 1000)

# Plot the function
plt.plot(x_values, f(x_values), label='$f(x) = x^2 - 4$')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--', label='y=0')

# Initial guess
initial_guess = 2.0
plt.scatter(initial_guess, f(initial_guess), color='red', marker='o', label='Initial Guess')

# Apply Newton-Raphson method
root, iterations = newton_raphson(initial_guess)
plt.scatter(root, f(root), color='green', marker='o', label=f'Root (after {iterations} iterations)')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Newton-Raphson Method for Finding Root of $f(x) = x^2 - 4$')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


# In[ ]:




