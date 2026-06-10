
END TO END IMAGE CLASSIFICATION PROJECT

Dataset
Fashion-MNIST Dataset

Number of Classes
10

Image Size
28 x 28

Architecture

Conv2D 32
BatchNormalization
MaxPooling

Conv2D 64
BatchNormalization
MaxPooling

Conv2D 128
BatchNormalization
MaxPooling

Dense 256
Dropout 0.5

Dense 128
Dropout 0.3

Output Layer Softmax

Regularization

Batch Normalization
Dropout
Early Stopping
ReduceLROnPlateau

Augmentation

Random Flip
Random Rotation
Random Zoom
Random Translation

Optimizer

Adam

Loss Function

Sparse Categorical Crossentropy

Final Test Accuracy

0.8778
