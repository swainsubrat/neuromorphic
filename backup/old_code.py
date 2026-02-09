# This script requires PyTorch, TorchVision, and snnTorch.
# You can install them using pip:
# pip install torch torchvision snntorch matplotlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ---- 1. Data Loading and Preprocessing ----
# As in the paper, we will use the MNIST dataset.

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)), # MNIST mean and std
    transforms.Lambda(lambda x: x.view(28*28)) # Flatten the images
])

# Load datasets
mnist_train = datasets.MNIST("data", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST("data", train=False, download=True, transform=transform)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)


# ---- 2. Memristor Device Model ----
# This class simulates the behavior of the physical device based on its
# measured LTP/LTD characteristics.
# YOU WILL REPLACE THE PARAMETERS HERE WITH YOUR OWN MATERIAL'S DATA.

class MemristorModel:
    """
    Models the non-ideal, non-linear weight update of a memristor synapse.
    The update equations are taken from the provided research paper (Fig 6c).
    """
    def __init__(self, g_min=0.0, g_max=1.0, p_max=100):
        # Conductance range (normalized)
        self.g_min = g_min
        self.g_max = g_max
        # Max pulses to go from g_min to g_max
        self.p_max = p_max

        # Non-linearity factors from the paper for the WTe2 device
        # You will need to determine these for your own material.
        self.nl_ltp = 2.49
        self.nl_ltd = -1.76

        # Calculate dependent parameter 'B' based on 'A' (non-linearity)
        # to ensure the function spans the full conductance range.
        self.b_ltp = (self.g_max - self.g_min) / (1 - np.exp(-self.p_max / self.nl_ltp))
        self.b_ltd = (self.g_max - self.g_min) / (1 - np.exp(-self.p_max / abs(self.nl_ltd)))


    def _update_conductance(self, p, is_ltp):
        """Calculates new conductance based on pulse number 'p'."""
        if is_ltp:
            # Equation for LTP
            g = self.b_ltp * (1 - np.exp(-p / self.nl_ltp)) + self.g_min
        else:
            # Equation for LTD
            g = -self.b_ltd * (1 - np.exp(-(self.p_max - p) / abs(self.nl_ltd))) + self.g_max
        return np.clip(g, self.g_min, self.g_max)

    def _g_to_p(self, g, is_ltp):
        """Inverse function: calculates the number of pulses to reach conductance 'g'."""
        if is_ltp:
            # Inverse of the LTP equation
            val = (g - self.g_min) / self.b_ltp
            if val >= 1.0: return self.p_max # Avoid log(negative)
            p = -self.nl_ltp * np.log(1 - val)
        else:
            # Inverse of the LTD equation
            val = (self.g_max - g) / self.b_ltd
            if val >= 1.0: return self.p_max
            p = self.p_max + abs(self.nl_ltd) * np.log(1 - val)
        return np.clip(p, 0, self.p_max)

    def update_weights(self, weights, gradients, learning_rate=0.01):
        """
        Custom weight update function that simulates the memristor.
        It converts the ideal gradient update into a number of pulses,
        then finds the new conductance from the device model.
        """
        with torch.no_grad():
            # Ideal weight update from standard backpropagation
            ideal_update = -learning_rate * gradients
            
            # --- This is the core of the simulation ---
            # 1. Get the current state (number of pulses) for each weight
            current_p_ltp = self._g_to_p(weights.cpu().numpy(), is_ltp=True)
            current_p_ltd = self._g_to_p(weights.cpu().numpy(), is_ltp=False)

            # 2. Determine whether to apply LTP or LTD pulses based on gradient sign
            is_ltp_pulse = (ideal_update > 0).cpu().numpy()
            is_ltd_pulse = ~is_ltp_pulse

            # 3. Convert the ideal update magnitude into a number of pulses
            # This is a simple linear mapping. More complex mappings could be used.
            num_pulses = np.abs(ideal_update.cpu().numpy()) * self.p_max

            # 4. Calculate the new pulse number
            new_p = np.where(is_ltp_pulse, current_p_ltp + num_pulses, current_p_ltd - num_pulses)
            new_p = np.clip(new_p, 0, self.p_max)

            # 5. Get the new conductance (weight) from the device model
            new_weights_ltp = self._update_conductance(new_p, is_ltp=True)
            new_weights_ltd = self._update_conductance(new_p, is_ltp=False)

            final_new_weights = np.where(is_ltp_pulse, new_weights_ltp, new_weights_ltd)

            return torch.tensor(final_new_weights, dtype=weights.dtype, device=weights.device)


# ---- 3. Neural Network Definition ----
# A standard Multi-Layer Perceptron (MLP), as described in the paper.
# Note: The paper crops to 20x20, we use the full 28x28 for simplicity here.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)
        # We clamp the weights to the device's normalized conductance range
        self.fc1.weight.data.uniform_(0, 1)
        self.fc2.weight.data.uniform_(0, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# ---- 4. Training and Testing Loop ----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
memristor_model = MemristorModel()
loss_fn = nn.CrossEntropyLoss()

# We don't use a standard optimizer like Adam. The update logic is in the MemristorModel.
# The training loop will manually apply the memristor-based updates.

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss = loss_fn(output, target)
        
        # Standard backpropagation to get gradients
        model.zero_grad()
        loss.backward()

        # Custom weight update using the memristor model
        new_w1 = memristor_model.update_weights(model.fc1.weight, model.fc1.weight.grad)
        model.fc1.weight.data = new_w1
        
        new_w2 = memristor_model.update_weights(model.fc2.weight, model.fc2.weight.grad)
        model.fc2.weight.data = new_w2

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

# ---- 5. Main Execution ----
num_epochs = 5
accuracies = []
for epoch in range(1, num_epochs + 1):
    train(epoch)
    acc = test()
    accuracies.append(acc)

print("Simulation finished.")

# Plot accuracy
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), accuracies, marker='o')
plt.title('MNIST Recognition Accuracy with Simulated Memristor')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.grid(True)
plt.show()
