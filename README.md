# Histopathology_Cancer_Detection

## Introduction
#### Pathologists face a substantial increase in workload and complexity of histopathologic cancer diagnosis due to the advent of personalized medicine. Traditionally, pathologists analyze tissue samples under a microscope to identify cancer cells. However, this process is time-consuming, subjective, and prone to errors. The emergence of deep learning, specifically Convolutional Neural Networks (CNNs), has revolutionized cancer detection in histopathology images. This report focuses on how CNN find evidence of cancer in images and also by using autoencoders an unsupervised deep learning model, which would improve the accuracy.

## Data Preparation and Pre Processing
#### Collected the data from a Kaggle Competition dataset. This data consists of test and train data which we have used to train our model. The dataset size is 6.3 GB, used google drive to warehouse the dataset and then by using the colab environment extracted the data to train models.
#### The training dataset consists of 57,458 values, whereas the test dataset consists of 220,025 values. In the training dataset, the value counts of images having cancer and images with no cancer are as follows.
#### Performed transformations to the data by using techniques like flipping, rotating and normalization. This is to ensure the diversity of training data is increased and to prevent overfitting the model to the training data. They created a batch size of 128 and 10 per cent of training data to use for validation. Prepared data loaders for training and validation.

## Modelling

### Convolutional Neural Network

#### Built a Convolutional Neural network (CNN) with 5 convolutional layers with an increasing number of filters 32 to 512. Each convolutional layer is followed by batch normalization, ReLU activation function, and max pooling operation for downsampling. The output of the convolutional layers is flattened and fed into a fully connected neural network with three linear layers and ReLU activation functions. The last layer uses a sigmoid activation function to output a binary classification result. 

#### A validation area under the curve of 96.89% is achieved by the CNN model. Where these are the image predictions this CNN performed on the Histopathology test cancer dataset.

### Autoencoder Model

#### An Autoencoder model with 3 colour channels is used, it is passed through a series of convolutional layers with an increasing number of filters (32,64,128,256, and 512) and max pooling layers, resulting in a compressed representation of the input. The decoder then takes this compression representation and passes it through a series of transposed convolutional layers with decreasing numbers of filters (512, 256, 128, 64, and 32) and no pooling, resulting in a reconstructed output image with the same dimensions as the input. The Sigmoid function is applied to the output of the final layer of the decoder, and the forward () method defines the flow of the data through the model.

#### With Unsupervised learning, the autoencoder model has helped in image processing by denoising and reconstructing the images in the test dataset.

## Challenges Faced and Future Improvements
#### Some challenges experienced:
#### •	The RAM of the colab processor was not enough to run over 10 epochs. The results would have been better if the model is trained for 20 epochs. 
#### •	The Reconstructed images were stored on the drive, these reconstructed images will be more useful in detecting the evidence of cancer.
#### •	This was not possible because of an issue with the path name access.

## Conclusion
#### This project discusses creating a CNN model to identify the evidence of Cancer in the Histopathology images. The CNN has performed well resulting in a 97% Area Under the Curve. It also goes on to show how the Unsupervised learning model Autoencoder has helped to enhance the images. From the results, we can observe that there was a lot of noise in the dataset images. The Autoencoder has denoised the images and reconstructed them so that the pathologists would have a better visual of the evidence of cancer-affected cells.

