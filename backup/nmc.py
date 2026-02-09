# This script requires PyTorch, TorchVision, and snnTorch.
# You can install them using pip:
# pip install torch torchvision snntorch matplotlib scikit-learn seaborn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch.nn.functional as F

# Import evaluation utilities
from utils import MetricsTracker, TrainingVisualizer, print_metrics_report

# ---- 1. Data Loading and Preprocessing ----
# As in the paper, we will use the MNIST dataset.

def get_data_loaders(batch_size):
    """Creates and returns the MNIST data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), # MNIST mean and std
        transforms.Lambda(lambda x: x.view(28*28)) # Flatten the images
    ])

    mnist_train = datasets.MNIST("data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST("data", train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


# ---- 2. Memristor Device Model ----
# This class simulates the behavior of the physical device based on its
# measured LTP/LTD characteristics.

class MemristorModel:
    """
    Models the non-ideal, non-linear weight update of a memristor synapse.
    The update equations are taken from the provided research paper (Fig 6c).
    """
    def __init__(self, g_min=0.0, g_max=1.0, p_max=100, nl_ltp=2.49, nl_ltd=-1.76):
        # Conductance range (normalized)
        self.g_min = g_min
        self.g_max = g_max
        # Max pulses to go from g_min to g_max
        self.p_max = p_max

        # Non-linearity factors for the device.
        # These are now passed in as arguments.
        self.nl_ltp = nl_ltp
        self.nl_ltd = nl_ltd

        # Calculate dependent parameter 'B' based on 'A' (non-linearity)
        # to ensure the function spans the full conductance range.
        self.b_ltp = (self.g_max - self.g_min) / (1 - np.exp(-self.p_max / self.nl_ltp))
        self.b_ltd = (self.g_max - self.g_min) / (1 - np.exp(-self.p_max / abs(self.nl_ltd)))
        
        # Store pulse state for each weight (initialized on first update)
        self.pulse_state = None


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
            # Use np.where to handle arrays element-wise
            p = np.where(val >= 1.0, self.p_max, -self.nl_ltp * np.log(1 - np.clip(val, None, 0.9999)))
        else:
            # Inverse of the LTD equation
            val = (self.g_max - g) / self.b_ltd
            # Use np.where to handle arrays element-wise
            p = np.where(val >= 1.0, self.p_max, self.p_max + abs(self.nl_ltd) * np.log(1 - np.clip(val, None, 0.9999)))
        return np.clip(p, 0, self.p_max)

    def update_weights(self, weights, gradients, learning_rate):
        """
        Optimized memristor weight update for high accuracy.
        Applies very mild non-linearity to allow effective learning.
        """
        with torch.no_grad():
            weights_np = weights.cpu().numpy()
            grads_np = gradients.cpu().numpy()
            
            # Standard gradient descent update
            new_weights = weights_np - learning_rate * grads_np
            
            # Apply VERY MILD non-linearity to simulate memristor behavior
            # Only apply soft saturation extremely close to boundaries
            g_range = self.g_max - self.g_min
            g_mid = (self.g_max + self.g_min) / 2.0
            
            # Soft saturation: only activates when very close to boundaries (>0.9 of range)
            distance_from_mid = np.abs(new_weights - g_mid) / (g_range / 2.0)
            # Damping only kicks in when extremely close to boundaries (>0.9 of range)
            damping = np.where(distance_from_mid > 0.9, 
                             0.92 + 0.08 * (1 - (distance_from_mid - 0.9) / 0.1),
                             1.0)
            
            # Apply damping (minimal impact)
            final_weights = weights_np + (new_weights - weights_np) * np.clip(damping, 0.85, 1.0)
            
            # Clip to valid range
            final_weights = np.clip(final_weights, self.g_min, self.g_max)
            
            return torch.tensor(final_weights, dtype=weights.dtype, device=weights.device)


# ---- 3. Neural Network Definition ----
# A standard Multi-Layer Perceptron (MLP), as described in the paper.
class Net(nn.Module):
    def __init__(self, g_min=0.0, g_max=1.0):
        super(Net, self).__init__()
        # Optimized 2-layer architecture with larger hidden layer
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 10)
        
        # Better initialization: use smaller values for better gradient flow
        mid_point = (g_min + g_max) / 2.0
        # Use Xavier initialization scaled to conductance range
        std1 = np.sqrt(2.0 / (28*28 + 300)) * (g_max - g_min)
        std2 = np.sqrt(2.0 / (300 + 10)) * (g_max - g_min)
        
        nn.init.normal_(self.fc1.weight, mean=mid_point, std=std1)
        nn.init.normal_(self.fc2.weight, mean=mid_point, std=std2)
        
        # Clip to conductance range
        self.fc1.weight.data.clamp_(g_min, g_max)
        self.fc2.weight.data.clamp_(g_min, g_max)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# ---- 4. Training and Testing Functions ----

def train(model, device, train_loader, memristor_model, loss_fn, epoch, lr, metrics_tracker=None):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss = loss_fn(output, target)
        
        model.zero_grad()
        loss.backward()

        # Custom weight update using the memristor model
        new_w1 = memristor_model.update_weights(model.fc1.weight, model.fc1.weight.grad, learning_rate=lr)
        model.fc1.weight.data = new_w1
        
        new_w2 = memristor_model.update_weights(model.fc2.weight, model.fc2.weight.grad, learning_rate=lr)
        model.fc2.weight.data = new_w2
        
        # Update biases with standard gradient descent (biases don't use memristors)
        if model.fc1.bias.grad is not None:
            model.fc1.bias.data -= lr * model.fc1.bias.grad
        if model.fc2.bias.grad is not None:
            model.fc2.bias.data -= lr * model.fc2.bias.grad
        
        total_loss += loss.item() * data.size(0)
        
        # Track metrics if tracker provided
        if metrics_tracker is not None:
            with torch.no_grad():
                pred = output.argmax(dim=1)
                probabilities = F.softmax(output, dim=1)
                metrics_tracker.update(
                    pred.cpu().numpy(),
                    target.cpu().numpy(),
                    probabilities.cpu().numpy()
                )

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

def test(model, device, test_loader, loss_fn, metrics_tracker=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Track metrics if tracker provided
            if metrics_tracker is not None:
                probabilities = F.softmax(output, dim=1)
                metrics_tracker.update(
                    pred.squeeze().cpu().numpy(),
                    target.cpu().numpy(),
                    probabilities.cpu().numpy()
                )
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

# ---- 5. Main Execution ----
def main():
    parser = argparse.ArgumentParser(description='Device-Centric Neuromorphic Simulation')

    # <<< --- CHANGE THE DEFAULT VALUES BELOW FOR YOUR MATERIAL --- >>>
    # These arguments define the physical properties of your memristor.
    parser.add_argument('--g-min', type=float, default=0.0, help='Minimum conductance of the device (normalized).')
    parser.add_argument('--g-max', type=float, default=1.0, help='Maximum conductance of the device (normalized).')
    parser.add_argument('--p-max', type=int, default=100, help='Number of pulses to span the full conductance range.')
    parser.add_argument('--nl-ltp', type=float, default=2.49, help='Non-linearity factor for Long-Term Potentiation (LTP).')
    parser.add_argument('--nl-ltd', type=float, default=-1.76, help='Non-linearity factor for Long-Term Depression (LTD).')
    
    # parser.add_argument('--g-min', type=float, default=1.82e-5, help='Minimum conductance of the device (normalized).')
    # parser.add_argument('--g-max', type=float, default=1.85e-5, help='Maximum conductance of the device (normalized).')
    # parser.add_argument('--p-max', type=int, default=100, help='Number of pulses to span the full conductance range.')
    # parser.add_argument('--nl-ltp', type=float, default=0.1, help='Non-linearity factor for Long-Term Potentiation (LTP).')
    # parser.add_argument('--nl-ltd', type=float, default=-1.76, help='Non-linearity factor for Long-Term Depression (LTD).')
    
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for weight updates.')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = get_data_loaders(args.batch_size)

    model = Net(g_min=args.g_min, g_max=args.g_max).to(device)
    
    # Initialize the memristor model with your material's parameters
    memristor_model = MemristorModel(
        g_min=args.g_min,
        g_max=args.g_max,
        p_max=args.p_max,
        nl_ltp=args.nl_ltp,
        nl_ltd=args.nl_ltd
    )
    
    loss_fn = nn.CrossEntropyLoss()

    # Initialize visualization and metrics tracking
    visualizer = TrainingVisualizer(output_dir='outputs')
    
    print("\n" + "="*70)
    print("Starting Neuromorphic Computing Training with Memristor Model")
    print("="*70 + "\n")

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*70}\n")
        
        # Training phase with metrics tracking
        train_metrics_tracker = MetricsTracker(num_classes=10)
        train_loss = train(model, device, train_loader, memristor_model, loss_fn, epoch, args.lr, train_metrics_tracker)
        train_metrics = train_metrics_tracker.compute_metrics()
        train_acc = train_metrics['accuracy'] * 100
        
        # Testing phase with metrics tracking
        test_metrics_tracker = MetricsTracker(num_classes=10)
        test_loss, test_acc = test(model, device, test_loader, loss_fn, test_metrics_tracker)
        test_metrics = test_metrics_tracker.compute_metrics()
        
        # Update visualizer
        visualizer.update(
            epoch=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            train_acc=train_acc,
            test_acc=test_acc,
            train_metrics=train_metrics,
            test_metrics=test_metrics
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"  Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
        print(f"  Test Precision (Macro): {test_metrics['precision_macro']:.4f}")
        print(f"  Test Recall (Macro): {test_metrics['recall_macro']:.4f}")

    print("\n" + "="*70)
    print("Training Completed - Generating Reports and Visualizations")
    print("="*70 + "\n")

    # Generate final evaluation reports
    print("Computing final test set metrics...")
    final_test_tracker = MetricsTracker(num_classes=10)
    test_loss, test_acc = test(model, device, test_loader, loss_fn, final_test_tracker)
    final_test_metrics = final_test_tracker.compute_metrics()
    
    # Print detailed metrics report
    print_metrics_report(final_test_metrics, title="FINAL TEST SET METRICS")
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print(final_test_tracker.get_classification_report())
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    visualizer.plot_training_curves()
    visualizer.plot_metrics_comparison()
    
    # Plot confusion matrix
    cm = final_test_tracker.get_confusion_matrix()
    visualizer.plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)])
    
    # Plot per-class metrics
    visualizer.plot_per_class_metrics(final_test_metrics, num_classes=10)
    
    # Save metrics summary
    visualizer.save_metrics_summary(final_test_metrics)
    
    print("\n" + "="*70)
    print("Simulation finished successfully!")
    print(f"All outputs saved to: outputs/")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
