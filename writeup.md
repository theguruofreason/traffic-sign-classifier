**Traffic Sign Recognition**

The code is [here](https://github.com/theguruofreason/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb).

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./test_images/001.jpg "Traffic Sign 1"
[image2]: ./test_images/002.jpg "Traffic Sign 2"
[image3]: ./test_images/003.jpg "Traffic Sign 3"
[image4]: ./test_images/004.jpg "Traffic Sign 4"
[image5]: ./test_images/005.jpg "Traffic Sign 5"
[image6]: ./test_images/006.jpg "Traffic Sign 6"
[image7]: ./sign_original.png "Traffic Sign Original"
[image8]: ./sign_normalized.png "Traffic Sign Normalized"
[image9]: ./data_histogram.png "Histogram of Training Data"

---

**Dataset Exploration:**

The dataset contains 60,000 images of german traffic signs of 43 different classifications and their associated labels. All images are 32x32x3. The dataset also contains a validation set and test set for convenience. A single example of the dataset is visualized and its label is printed alongside it to verify the integrity of the data. The training set is 34,799 images, the validation set is 4410 images, and the test is 12,630 images.

Example Image:

![alt text][image7]

Histogram of Training Data:

![alt text][image9]

---

   **Design and Test a Model Architecture:**

The images were preprocessed by cv2 normalization as shown:
	
original
![alt text][image7]
normalized
![alt text][image8]

The normalization will likely help the neural network perform better.

The model I chose uses the standard LeNet architecture but also introduces a dropout layer before the second fully connected layer with a 70% keep probability. This results in significantly reduced overfitting which brings the validation accuracy much closer to the training accuracy. Initially, I did not include the dropout layer, but was achieving validation accuracies of only 88%. The addition of the dropout layer raised my validation accuracy significantly, and made me confident that the model would perform well on the test set.

The layers of the convolutional neural network  are as follows:

1. a 3x3 convolution
2. a Relu activation
3. a 2x2 max pooling
4. a 3x3 convolution
5. a Relu activation
6. a 2x2 max pooling
7. a flattened, fully connected layer
8. a Relu activation
9. a 70% keep dropout
10. a fully connected layer
11. a Relu activation
12. a final fully connected layer

To train the model, I used a learning rate of .001, a batch size of 200, and 15 epochs. I used the Adam optimizer because it is very robust, and a softmax activation to classify the data. As usual, the Adam optimizer was used to reduce cross entropy during training. 

This model produced fantastic results:

* Training accuracy: 99.6%
* Validation accuracy: 93.5%
* Test accuracy: 93%
 
The LeNet architecture is a good choice because the convolutional and pooling layers are useful for grouping sections of the image together, allowing it to determine lines and arcs such as the edges of the signs as well as complex shapes like the images of cars or deer on the signs.
 
---
 
**Test a Model on New Images**

Here are six German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5] ![alt text][image6]

These images are likely vector-drawn mockups of real life signs. They are not cropped images of signs photographed in the world. It's possible that they are oriented differently from much of the training data, and possibly also have different color saturations, sizes, or centering.

My model was able to successfully classify 6 out of 7 of the new traffic signs, earning an accuracy of 83.3%. The top five softmax probabilities were calculated, and the top probability was always over 99.9%. The model incorrectly identified the "no vehicles" sign as "no vehicles over 3.5 metric tons". These signs are very similar in shape and color, so it's understandable that this sign was misclassified. It is unfortunate that the model was over 99% certain about this classification.

This could still stand to be improved. If I were to improve it, I would add a greyscale layer to the images to add more data to train with. It's also possible adding noise to the images will reduce the clarity of background shapes, causing the model to more easily recognize the sign instead of the sign and the background shapes.