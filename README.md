##   Facial Expression Recognition Using CNN
## Problem Statement

One of the important ways humans display emotions is through facial expressions which are a very important part of communication. Though nothing is said verbally, there is much to be understood about the messages we send and receive through the use of nonverbal communication. Facial expressions convey non verbal cues, and they play an important role in interpersonal relations
Emotion detection has always been an easy task for humans, but achieving the same task with a computer algorithm is quite challenging. With the recent advancement in computer vision and machine learning, it is possible to detect emotions from images. 

This project aims to use Convolutional Neural Networks (CNN) and Transfer Learning models for facial expression recognition. The input into our system is an image; then, we use CNN to predict the facial expression label which should be one these labels: anger, happiness, fear,sad, surprise disgust and neutral.

### Executive Summary


### Convolutional Neural Networks
Similar to how a child learns to recognise objects, we need to show an algorithm millions of pictures before it is be able to generalize the input and make predictions for images it has never seen before.

Computers ‘see’ in a different way than we do. Their world consists of only numbers. Every image can be represented as 2-dimensional arrays of numbers, known as pixels.

But the fact that they perceive images in a different way, doesn’t mean we can’t train them to recognize patterns, like we do. We just have to think of what an image is in a different way.

CNNs have two components:

<b>1. Feature extraction part</b>
- Conv2D: Convolution is performed on the input data with the use of a filter or kernel (these terms are used interchangeably) to then produce a feature map.
-  All the feature maps and put them together as the final output of the convolution layer.
-  For any kind of neural network to be powerful, it needs to contain non-linearity. Pass the result of the convolution operation through relu activation function
- Stride specifies how much we move the convolution filter at each step. This also makes the resulting feature map smaller since we are skipping over potential locations.
- MaxPooling2D: After a convolution operation, pooling is performed to reduce the dimensionality. This enables us to reduce the number of parameters, which both shortens the training time and combats overfitting.
- After the convolution + pooling layers we add a couple of fully connected layers to wrap up the CNN architecture.


<b>2.  The Classification part</b>
- both convolution and pooling layers are 3D volumes, but a fully connected layer expects a 1D vector of numbers. So we flatten the output of the final pooling layer to a vector and that becomes the input to the fully connected layer. Flattening is simply arranging the 3D volume of numbers into a 1D vector, nothing fancy happens here.
- Softmax is frequently appended to the last layer of an image classification network 
- Cross entropy loss is usually the loss function for such a multi-class classification problem. 

### Transfer Learning

Transfer learning is a concept according to which we can transfer the learning of other pre-trained models to our data. Instead of training our own custom neural network, we can use other popular pre-trained models and pass our data to those models and ultimately get the features for our images.

Inherently, convolution layer generates features for the images. It applies convolution operation on each pixel of the images and ultimately generates ’n’ dimensional array which are nothing but learnt features of the images.

The final features of the image which we get at the end of convolution neural network is known as bottleneck features. These bottleneck features are the learnt features of the images which are then feed to the MLP which acts as a top-model. This MLP then reduces loss function and updates the weights in MLP and kernels/filters in CNN.
Now, for our task we have chosen MobileNet and ResNet50 pre-trained neural network to generate bottleneck features. These neural networks are well trained on image-net dataset which contains millions of images.
Following pre-processing steps are performed:
- Converting images to gray-scale.
- Detect face in the image using OpenCV HAAR Cascade.
- Crop the image to the face.
- Resize the image


<b>Model Evaluation</b>

Compile the model with categorical_crossentropy as the loss function and using Adam optimizer. Use accuracy as the metrics for validation.
