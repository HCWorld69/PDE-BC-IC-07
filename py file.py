
# Import pytorch and other libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the PDE function
def pde(u, r, theta):
  # Use torch.autograd to compute the second derivatives of u with respect to r and theta
  u_rr = torch.autograd.grad(torch.autograd.grad(u, r, create_graph=True)[0], r)[0]
  u_r = torch.autograd.grad(u, r, create_graph=True)[0]
  u_tt = torch.autograd.grad(torch.autograd.grad(u, theta, create_graph=True)[0], theta)[0]
  # Return the PDE equation
  return u_rr + u_r / r + u_tt / (r ** 2)

# Define the boundary condition function
def bc(u, r, theta):
  # Use torch.where to assign the boundary values of u according to theta
  u_bc = torch.where((0 <= theta) & (theta <= np.pi), 20 * torch.ones_like(u), torch.zeros_like(u))
  # Return the boundary condition equation
  return u - u_bc

# Define the linear neural network model
class LinearNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(LinearNN, self).__init__()
    # Define the linear layers
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, output_size)
    # Define the activation function
    self.relu = nn.ReLU()
  
  def forward(self, x):
    # Pass the input through the linear layers and the activation function
    out = self.linear1(x)
    out = self.relu(out)
    out = self.linear2(out)
    return out

# Define the input size, hidden size, and output size
input_size = 2 # r and theta
hidden_size = 10 # arbitrary choice
output_size = 1 # u

# Create an instance of the linear neural network model
model = LinearNN(input_size, hidden_size, output_size)

# Define the loss function and the optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define the number of epochs and the batch size
epochs = 1000 # arbitrary choice
batch_size = 64 # arbitrary choice

# Define the domain and the range of r and theta
r_min = 1
r_max = 2
theta_min = 0
theta_max = 2 * np.pi

# Generate random samples of r and theta
r = torch.rand(batch_size) * (r_max - r_min) + r_min
theta = torch.rand(batch_size) * (theta_max - theta_min) + theta_min

# Reshape r and theta to match the input size
r = r.view(-1, 1)
theta = theta.view(-1, 1)

# Concatenate r and theta to form the input tensor
x = torch.cat((r, theta), dim=1)

# Train the model
for epoch in range(epochs):
  # Zero the gradients
  optimizer.zero_grad()
  # Forward pass
  u = model(x)
  # Compute the loss
  loss = loss_fn(pde(u, r, theta), torch.zeros_like(u)) + loss_fn(bc(u, r, theta), torch.zeros_like(u))
  # Backward pass
  loss.backward()
  # Update the parameters
  optimizer.step()
  # Print the loss every 100 epochs
  if (epoch + 1) % 100 == 0:
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Plot the solution
# Create a meshgrid of r and theta
r_plot = np.linspace(r_min, r_max, 100)
theta_plot = np.linspace(theta_min, theta_max, 100)
r_mesh, theta_mesh = np.meshgrid(r_plot, theta_plot)
# Reshape the meshgrid to match the input size
r_mesh = r_mesh.reshape(-1, 1)
theta_mesh = theta_mesh.reshape(-1, 1)
# Concatenate the meshgrid to form the input tensor
x_plot = np.concatenate((r_mesh, theta_mesh), axis=1)
# Convert the input tensor to torch tensor
x_plot = torch.from_numpy(x_plot).float()
# Predict the output using the model
u_plot = model(x_plot)
#