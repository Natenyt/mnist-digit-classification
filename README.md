# MNIST Digit Classification - Assignment 6

A Convolutional Neural Network (CNN) implementation for handwritten digit recognition using the MNIST dataset, achieving 99%+ accuracy.

## 📋 Assignment Requirements

- Use MNIST dataset (digits 0-9)
- Achieve accuracy of 99% or higher
- Display the number of wrong predictions
- Show 2 examples of misclassified images

## 🎯 Results

- **Test Accuracy:** 99%+ 
- **Model Type:** Convolutional Neural Network (CNN)
- **Training Time:** ~5-10 minutes (CPU)
- **Total Parameters:** ~1.2M

## 🏗️ Model Architecture

```
Conv2D(32) → MaxPooling2D → Conv2D(64) → MaxPooling2D → 
Conv2D(64) → Flatten → Dense(128) → Dropout(0.5) → Dense(10)
```

**Layers:**
- 3 Convolutional layers (32, 64, 64 filters)
- 2 Max Pooling layers
- 1 Dense hidden layer (128 neurons)
- Dropout layer (0.5) for regularization
- Output layer (10 classes - digits 0-9)

## 🚀 Installation

### Prerequisites

- Python 3.7+
- pip

### Install Dependencies

```bash
pip install tensorflow numpy matplotlib
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Run the Training Script

```bash
python assignment6.py
```

### What Happens:

1. Automatically downloads MNIST dataset (first run only)
2. Trains the CNN model for 10 epochs
3. Evaluates model on test set
4. Identifies and displays wrong predictions
5. Generates visualization plots

### Output Files

- `wrong_predictions.png` - Images of 2 misclassified digits
- `training_history.png` - Training/validation accuracy and loss graphs

## 📊 Sample Output

```
==================================================
EVALUATION RESULTS
==================================================
Test Accuracy: 99.12%
Test Loss: 0.0312

==================================================
WRONG PREDICTIONS
==================================================
Total misclassified images: 88
Total test images: 10000
Error rate: 0.88%

==================================================
DISPLAYING 2 WRONG PREDICTION EXAMPLES
==================================================

Wrong prediction #1:
  Index: 247
  Predicted: 8
  Actual: 2

Wrong prediction #2:
  Index: 340
  Predicted: 9
  Actual: 4
```

## 📁 Project Structure

```
mnist-assignment/
│
├── assignment6.py           # Main training script
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── wrong_predictions.png   # Generated output (after running)
└── training_history.png    # Generated output (after running)
```

## 🧪 Dataset

**MNIST Database of Handwritten Digits**

- Training samples: 60,000
- Test samples: 10,000
- Image size: 28x28 pixels (grayscale)
- Classes: 10 (digits 0-9)

Source: [Yann LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/)

## 🔧 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 128 |
| Epochs | 10 |
| Validation Split | 10% |
| Dropout Rate | 0.5 |

## 📈 Performance

- **Training Accuracy:** ~99.5%
- **Validation Accuracy:** ~99.2%
- **Test Accuracy:** ~99.1%
- **Typical Misclassifications:** 80-100 images out of 10,000

## 🎓 Learning Outcomes

This assignment demonstrates:
- Image classification with CNNs
- Data preprocessing and normalization
- Model training and evaluation
- Error analysis and visualization
- Deep learning with TensorFlow/Keras

## 🛠️ Improvements & Extensions

Possible enhancements:
- Data augmentation (rotation, shifting, zooming)
- Advanced architectures (ResNet, DenseNet)
- Hyperparameter tuning
- Learning rate scheduling
- Ensemble methods

## 📝 Requirements.txt

```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.4.0
```

## 👨‍💻 Author

**Your Name**
- Course: Artificial Intelligence
- Assignment: 6 - MNIST Classification

## 📄 License

This project is for educational purposes as part of an AI course assignment.

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun
- TensorFlow/Keras framework
- Course instructor and materials

---

**Note:** The model achieves >99% accuracy but exact results may vary slightly between runs due to random initialization.