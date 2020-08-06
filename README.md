{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##   Facial Expression Recognition Using CNN\n",
    "## Problem Statement\n",
    "\n",
    "One of the important ways humans display emotions is through facial expressions which are a very important part of communication. Though nothing is said verbally, there is much to be understood about the messages we send and receive through the use of nonverbal communication. Facial expressions convey non verbal cues, and they play an important role in interpersonal relations\n",
    "Emotion detection has always been an easy task for humans, but achieving the same task with a computer algorithm is quite challenging. With the recent advancement in computer vision and machine learning, it is possible to detect emotions from images. \n",
    "\n",
    "This project aims to use Convolutional Neural Networks (CNN) and Transfer Learning models for facial expression recognition. The input into our system is an image; then, we use CNN to predict the facial expression label which should be one these labels: anger, happiness, fear,sad, surprise disgust and neutral.\n",
    "\n",
    "### Executive Summary\n",
    "\n",
    "\n",
    "### Convolutional Neural Networks\n",
    "Similar to how a child learns to recognise objects, we need to show an algorithm millions of pictures before it is be able to generalize the input and make predictions for images it has never seen before.\n",
    "\n",
    "Computers ‘see’ in a different way than we do. Their world consists of only numbers. Every image can be represented as 2-dimensional arrays of numbers, known as pixels.\n",
    "\n",
    "But the fact that they perceive images in a different way, doesn’t mean we can’t train them to recognize patterns, like we do. We just have to think of what an image is in a different way.\n",
    "\n",
    "CNNs have two components:\n",
    "\n",
    "<b>1. Feature extraction part</b>\n",
    "- Conv2D: Convolution is performed on the input data with the use of a filter or kernel (these terms are used interchangeably) to then produce a feature map.\n",
    "-  All the feature maps and put them together as the final output of the convolution layer.\n",
    "-  For any kind of neural network to be powerful, it needs to contain non-linearity. Pass the result of the convolution operation through relu activation function\n",
    "- Stride specifies how much we move the convolution filter at each step. This also makes the resulting feature map smaller since we are skipping over potential locations.\n",
    "- MaxPooling2D: After a convolution operation, pooling is performed to reduce the dimensionality. This enables us to reduce the number of parameters, which both shortens the training time and combats overfitting.\n",
    "- After the convolution + pooling layers we add a couple of fully connected layers to wrap up the CNN architecture.\n",
    "\n",
    "\n",
    "<b>2.  The Classification part</b>\n",
    "- both convolution and pooling layers are 3D volumes, but a fully connected layer expects a 1D vector of numbers. So we flatten the output of the final pooling layer to a vector and that becomes the input to the fully connected layer. Flattening is simply arranging the 3D volume of numbers into a 1D vector, nothing fancy happens here.\n",
    "- Softmax is frequently appended to the last layer of an image classification network \n",
    "- Cross entropy loss is usually the loss function for such a multi-class classification problem. \n",
    "\n",
    "### Transfer Learning\n",
    "\n",
    "Transfer learning is a concept according to which we can transfer the learning of other pre-trained models to our data. Instead of training our own custom neural network, we can use other popular pre-trained models and pass our data to those models and ultimately get the features for our images.\n",
    "\n",
    "Inherently, convolution layer generates features for the images. It applies convolution operation on each pixel of the images and ultimately generates ’n’ dimensional array which are nothing but learnt features of the images.\n",
    "\n",
    "The final features of the image which we get at the end of convolution neural network is known as bottleneck features. These bottleneck features are the learnt features of the images which are then feed to the MLP which acts as a top-model. This MLP then reduces loss function and updates the weights in MLP and kernels/filters in CNN.\n",
    "Now, for our task we have chosen MobileNet and ResNet50 pre-trained neural network to generate bottleneck features. These neural networks are well trained on image-net dataset which contains millions of images.\n",
    "Following pre-processing steps are performed:\n",
    "- Converting images to gray-scale.\n",
    "- Detect face in the image using OpenCV HAAR Cascade.\n",
    "- Crop the image to the face.\n",
    "- Resize the image\n",
    "\n",
    "\n",
    "<b>Model Evaluation</b>\n",
    "\n",
    "Compile the model with categorical_crossentropy as the loss function and using Adam optimizer. Use accuracy as the metrics for validation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
