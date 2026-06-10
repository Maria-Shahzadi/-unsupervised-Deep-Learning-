FNN End-to-End Pipeline: Build a complete FNN in Keras on the MNIST dataset.
Design a 3-layer architecture with ReLU hidden activations and Softmax output. Compile
with Adam and categorical cross-entropy. Train for 20 epochs with a validation split and
EarlyStopping. Plot training vs validation loss and accuracy curves. Evaluate on the test
set and print the full classification report. Save the trained model, reload it, and run
inference on 5 unseen samples — display each image alongside its predicted and true
label