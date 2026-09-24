# Explain: The LLM (Qwen3-Coder-Plus) is asked to build a CNN. But its code does not use any convolutional layer.


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

# Demo input data
x_data = [[0.1, 0.2], [0.3, 0.7], [0.6, 0.8], [0.8, 0.4], [0.9, 0.5]]
y_data = [0, 0, 1, 1, 1]

# Convert to numpy arrays for preprocessing
x_np = np.array(x_data, dtype=np.float32)
y_np = np.array(y_data, dtype=np.float32)

# Normalize the input data
scaler = StandardScaler()
x_normalized = scaler.fit_transform(x_np)

# Convert to PyTorch tensors
x_tensor = torch.from_numpy(x_normalized)
y_tensor = torch.from_numpy(y_np).unsqueeze(1)

# Define the neural network architecture
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        # Single layer neural network (linear transformation followed by activation)
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

# Initialize the model
model = BinaryClassifier()

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 200
losses = []

print("Starting training...")
print("Epoch\tLoss")

for epoch in range(epochs):
    # Forward pass
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f"{epoch+1}\t{loss.item():.6f}")
    
    # Check for convergence (if loss change is very small)
    if len(losses) > 10:
        recent_losses = losses[-10:]
        if max(recent_losses) - min(recent_losses) < 1e-6:
            print(f"Converged at epoch {epoch+1}")
            break

print(f"\nTraining completed after {len(losses)} epochs")
print(f"Final loss: {losses[-1]:.6f}")

# Test the model with the training data
model.eval()
with torch.no_grad():
    predictions = model(x_tensor)
    predicted_classes = (predictions > 0.5).float()
    
print("\nResults:")
print("Input\t\tNormalized\tPredicted Prob\tPredicted Class\tActual Class")
for i in range(len(x_data)):
    original_input = x_data[i]
    normalized_input = x_normalized[i]
    prob = predictions[i].item()
    pred_class = int(predicted_classes[i].item())
    actual_class = int(y_data[i])
    print(f"{original_input}\t{normalized_input.round(3)}\t\t{prob:.3f}\t\t{pred_class}\t\t{actual_class}")

# Test with new data points
print("\nTesting with new data points:")
test_points = [[0.2, 0.3], [0.7, 0.6], [0.5, 0.9]]
test_points_np = np.array(test_points, dtype=np.float32)
test_points_normalized = scaler.transform(test_points_np)
test_points_tensor = torch.from_numpy(test_points_normalized)

with torch.no_grad():
    test_predictions = model(test_points_tensor)
    test_classes = (test_predictions > 0.5).float()
    
for i in range(len(test_points)):
    point = test_points[i]
    prob = test_predictions[i].item()
    pred_class = int(test_classes[i].item())
    print(f"Input: {point} -> Probability: {prob:.3f}, Predicted Class: {pred_class}")