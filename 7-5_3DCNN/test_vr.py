import torch

# Dummy MSE loss
mse_loss = torch.tensor(5.0)

# Simulate variance_regularization function (this is for testing purposes)
def variance_regularization(predictions, alpha=0.01):
    # Simulate NaN or valid output
    variance = torch.var(predictions)
    result = alpha * (1.0 / (variance + 1e-6))
    return result

# Test Case 1: Normal Case (No NaN in var_loss)
predictions_normal = torch.tensor([1.0, 2.0, 3.0])  # Example predictions
var_loss_normal = variance_regularization(predictions_normal)

if torch.isnan(var_loss_normal):
    print("Warning: NaN value detected in variance regularization loss. Using only MSE loss.")
    loss = mse_loss
else:
    loss = mse_loss + var_loss_normal
print("Test Case 1 - Loss:", loss.item())


# Test Case 2: NaN Case (Var loss is NaN)
predictions_nan = torch.tensor([float('nan'), 2.0, 3.0])
var_loss_nan = variance_regularization(predictions_nan)

if torch.isnan(var_loss_nan):
    print("Warning: NaN value detected in variance regularization loss. Using only MSE loss.")
    loss = mse_loss
else:
    loss = mse_loss + var_loss_nan
print("Test Case 2 - Loss:", loss.item())