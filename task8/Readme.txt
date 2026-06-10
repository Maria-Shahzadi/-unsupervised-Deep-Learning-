MLP Regularisation Experiment: Using a tabular dataset prone to overfitting (small
dataset with many features), build a deep MLP in Keras. First train with no regularisation
and demonstrate overfitting clearly through diverging loss curves. Then add Dropout
layers (rates 0.2 and 0.5) and retrain. Then replace Dropout with Batch Normalisation.
Finally combine both. Plot all four models' training vs validation curves side by side. Print
a final summary table of test accuracy and val_loss for all four configurations with a
written conclusion on which regularisation strategy worked best