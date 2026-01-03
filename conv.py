from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN (add channel dimension)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Dataset shapes:")
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# Train the model
print("\nTraining the model...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluate on test set
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Predict on all test images
predictions = model.predict(x_test, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

# Find misclassified images
wrong_indices = np.where(predicted_labels != y_test)[0]
num_wrong = len(wrong_indices)

print("\n" + "="*50)
print("WRONG PREDICTIONS")
print("="*50)
print(f"Total misclassified images: {num_wrong}")
print(f"Total test images: {len(y_test)}")
print(f"Error rate: {(num_wrong/len(y_test))*100:.2f}%")

# Show first 2 wrong predictions
if num_wrong >= 2:
    print("\n" + "="*50)
    print("DISPLAYING 2 WRONG PREDICTION EXAMPLES")
    print("="*50)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for i in range(2):
        idx = wrong_indices[i]
        ax = axes[i]
        
        # Display image
        ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"Predicted: {predicted_labels[idx]}\nActual: {y_test[idx]}", 
                     fontsize=12, color='red')
        ax.axis('off')
        
        print(f"\nWrong prediction #{i+1}:")
        print(f"  Index: {idx}")
        print(f"  Predicted: {predicted_labels[idx]}")
        print(f"  Actual: {y_test[idx]}")
    
    plt.tight_layout()
    plt.savefig('wrong_predictions.png', dpi=150, bbox_inches='tight')
    print("\nImages saved as 'wrong_predictions.png'")
    plt.show()
    
elif num_wrong == 1:
    print("\nOnly 1 wrong prediction found!")
    idx = wrong_indices[0]
    plt.figure(figsize=(5, 4))
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_labels[idx]}\nActual: {y_test[idx]}", 
              fontsize=12, color='red')
    plt.axis('off')
    plt.savefig('wrong_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Wrong prediction at index {idx}: Predicted {predicted_labels[idx]}, Actual {y_test[idx]}")
else:
    print("\nNo wrong predictions! Perfect accuracy!")

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Model Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("ASSIGNMENT COMPLETED!")
print("="*50)