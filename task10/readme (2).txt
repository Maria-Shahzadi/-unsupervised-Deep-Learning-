End-to-End Image Classification: Pick a dataset (CIFAR-10, Fashion-MNIST, or a
small Kaggle image dataset). Build a complete pipeline — load and normalise images →
apply Data Augmentation → build a CNN with at least 3 Conv+Pool blocks → add
Dropout and Batch Normalisation → compile with Adam → train with EarlyStopping and
ReduceLROnPlateau callbacks → evaluate with Confusion Matrix and Classification
Report → visualise a grid of misclassified samples with true and predicted labels → save
the final model and reload it to predict on 5 brand new images. Write a README.txt
summarising the dataset, architecture choices, regularisation decisions, augmentation
strategy, and final test accuracy achieved.