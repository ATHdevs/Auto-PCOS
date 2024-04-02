# Auto-PCOS

This pipeline involves training a binary classification model using transfer learning with a ResNetRS420 
base model. Here's a brief overview of the steps involved:
1. Image Preprocessing:
- Images are loaded and preprocessed using the preprocess_image function, which 
resizes the images to the specified dimensions (img_width, img_height) and applies 
preprocessing suitable for the ResNetRS420 model using 
tf.keras.applications.inception_v3.preprocess_input.

2. Data Preparation:
- The preprocessed images are collected along with their corresponding labels from the 
training, validation, and test datasets (train_df, val_df, test_df).
- Images are converted to numpy arrays to be fed into the model.
  
3. Base Model Initialization:
- The InceptionV3 model is loaded with the pre-trained ImageNet weights. Only the 
convolutional base of the model is included (include_top=False) as custom dense 
layers will be added for classification.

4. Model Architecture:
Custom classification layers are added on top of the base model:
- LayerNormalization layer to normalize the activations.
- Convolutional layer with 1024 filters and a ReLU activation function.
- MaxPooling layer to downsample the spatial dimensions.
- Dropout layer with a dropout rate of 0.3 to prevent overfitting.
- Flattening layer to convert the 2D feature maps into a 1D vector.
- Dense layer with 512 units and a ReLU activation function.
- Output dense layer with a single unit and a sigmoid activation function for binary classification.
  
5. Model Compilation:
- The model is compiled with the Adam optimizer and binary cross-entropy loss function. 
Binary accuracy is chosen as the evaluation metric.

6. Model Training:
- The model is trained on the training dataset (train_images, train_labels) for 50 epochs.
- Validation data (val_images, val_labels) is provided for monitoring the model's 
performance during training.

7. Monitoring Training Progress:
- The training progress is monitored using the history object returned by the fit method, 
which contains metrics like loss and accuracy on both training and validation datasets 
for each epoch.

8. Model Evaluation:
- After training, the model's performance can be evaluated on the test dataset 

(test_images, test_labels) using appropriate evaluation metrics.

This pipeline leverages transfer learning to utilize the pre-trained InceptionV3 model's feature 
extraction capabilities while fine-tuning the model for the specific binary classification task. It follows 
standard practices for training deep learning models, including data preprocessing, model 
construction, training, and evaluation


![Data](https://github.com/ATHdevs/Auto-PCOS/assets/147138099/17575967-bcce-4348-bf45-bf6001e286fb)


| Metric    | Score            |
|-----------|------------------|
| Accuracy  | 0.9052           |
| Precision | 0.9001           |
| Recall    | 0.9716           |
| F1 Score  | 0.9345           |


![image](https://github.com/ATHdevs/Auto-PCOS/assets/147138099/e7481a77-7de1-48d4-a6fb-4566d916f8a5)


![image](https://github.com/ATHdevs/Auto-PCOS/assets/147138099/29a50cb8-c4ca-4230-a0ca-4c68ca117aa8)
![image](https://github.com/ATHdevs/Auto-PCOS/assets/147138099/b52433a9-e8d3-4111-a604-707377e7a304)

