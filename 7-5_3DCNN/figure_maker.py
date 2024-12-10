import matplotlib.pyplot as plt

# Lists to store the training and validation loss values
training_losses = [
    18.4174, 18.0869, 17.7969, 17.6828, 17.3874, 17.2182, 17.1090, 16.9112, 16.6761,
    16.4810, 16.2812, 16.2608, 16.0817, 15.9374, 15.7683, 15.5320, 15.4512, 15.2216,
    14.9927, 14.8223
]
validation_losses = [
    18.1749, 17.9087, 17.6464, 17.5434, 17.3796, 17.1428, 17.2949, 17.1620, 17.1450,
    16.8573, 16.8960, 17.1626, 17.0421, 17.0449, 17.2012, 17.0439, 17.1015, 17.3808,
    17.4508, 17.4418
]

training_losses = [
    18.467853976840725, 17.72080452639729, 17.36734675142864, 17.060972822632266,
    16.790005987077162, 16.630015649814027, 16.44221516610315, 16.27274519712701,
    16.182539292390288, 16.088474047301776, 15.918263926885423, 15.805126216169622,
    15.707193947664441, 15.56053138048483, 15.432937022272426, 15.410738985288972,
    15.272272349053457, 15.055288358580214, 14.888686553405579, 14.831884499572757
]

validation_losses = [
    17.9011, 17.6110, 17.1620, 17.0855, 16.8246, 16.7784, 16.8835, 16.7080,
    16.5592, 16.3160, 16.5482, 16.5498, 16.6264, 16.4929, 16.4411, 16.4891,
    16.7727, 16.6673, 16.5984, 16.8390
]


# training_losses = [
#     18.5285, 17.6995, 17.3256, 16.9867, 16.7985, 16.6357,
#     16.5255, 16.3380, 16.1863, 16.0835
# ]
#
# # Validation Loss values for each epoch
# validation_losses = [
#     17.9419, 17.4194, 17.0988, 16.9615, 16.9129, 16.5557,
#     16.6668, 16.5826, 16.6815, 16.6197
# ]

# Number of epochs
epochs = list(range(1, 20 + 1))

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the training and validation loss
plt.plot(epochs, training_losses, label='Training Loss', marker='o', color='blue')
plt.plot(epochs, validation_losses, label='Validation Loss', marker='o', color='red')

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs (5% of dataset)')

plt.axhline(y=18.67, color='green', linestyle='--', label="MSE for predicting all 0")

# Add the vertical line for the minimum validation loss
min_val_loss = min(validation_losses)
min_val_epoch = validation_losses.index(min_val_loss) + 1  # +1 to match the epoch number
plt.axvline(x=min_val_epoch, color='orange', linestyle='--', label=f"Min Validation Loss at Epoch {min_val_epoch}")

plt.xticks(range(1, 21, 2))
plt.yticks([14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5])



plt.legend()

# Show the plot
plt.grid(True)
plt.show()
