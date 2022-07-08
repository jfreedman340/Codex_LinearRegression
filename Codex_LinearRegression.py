
# This is what I told the Codex AI.

"""
Create an example linear regressor using PyTorch
1. Create the x and y input and outputs
2. Create the weight vector as a tensor
3. Run a learning loop and for the regressor with SGD
4. Test the accuracy and print the results
"""

#This is what the Codex AI wrote.

import torch
import numpy as np
from matplotlib import pyplot as plt

# Create the x and y input and outputs
x = torch.linspace(0, 10, 100)
y = x + 3 * torch.rand(x.size())


# Create the weight vector as a tensor
w = torch.tensor([1.0], requires_grad=True)


# Run a learning loop and for the regressor with SGD
learning_rate = 0.005  # The learning rate for SGD


def forward(x):  # The forward pass of the model (prediction)
    return x * w


def loss(y, y_predicted):  # The loss function (MSE in this case)
    return ((y_predicted - y) ** 2).mean()


print("Progress:", "*" * 25)  # Print progress bar to track progress of training loop


for epoch in range(1, 100):  # Run 100 epochs of training loop

    y_pred = forward(x)  # Make prediction with current weight vector w

    l = loss(y, y_pred)  # Calculate the loss

    l.backward()  # Backpropagate the loss to calculate gradients

    with torch.no_grad():  # Update the weight vector w
        w -= learning_rate * w.grad

    w.requires_grad = True  # Set requires_grad to True for next iteration

    if epoch % 10 == 0:
        print("*", end='')  # Print progress bar to track progress of training loop

print("\n")
print("w =", w.item())  # Print the final weight vector value


# Test the accuracy and print the results
y_predicted = forward(x)  # Make prediction with current weight vector w
l = loss(y, y_predicted)  # Calculate the loss (MSE in this case)
print("Loss:", l.item())  # Print the final loss value


# Plot results of training loop and test accuracy of model on data set x and y
plt.plot(x.numpy(), y.numpy(), 'o')  # Plot data set x and y as scatter plot (o)
plt.plot(x.numpy(), y_predicted.detach().numpy())  # Plot the predicted values of x and y
plt.show()  # Show the plot