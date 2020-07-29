# Geran Traffic Sign Classifier


**The goal of this project is to train a CNN network that identifies German traffic signs.**

## Summary
Architecture: LeNet-5 with modified setting

Data source: [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

Accuracy: 
- Training: 99.5%
- Validation: 94.2%
- Test: 92%


## Data Set Summary & Exploration

#### Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy and matplotlib libraries to calculate and visualize the summary statistics of the traffic signs data set.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes in the data set is 43

### Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the distribution of the data set. It is a histogam chart showing how the data is spread accross different categories.
![](https://i.imgur.com/j7kyCHw.png)

From the plots, we can see that the distributions are similar in all data sets, so the training is fairly balanced.


## Design and Test a Model Architecture

### Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?

All the pixels of training images are coverted to floating number range from 0~1.
$$
\frac{X - X_{min}}{X_{max} - X_{min}}
$$

This step will recenter the mean and deviation of the data to almost 0.

The $X_{min}$ and $X_{max}$ from training set data are used on both validation and test set for preprocess consistency.

The preprocessed image looks identical to the original image but if we print out the max and min of the image, we can see that it is converted successfully.

![](https://i.imgur.com/5pxsomf.png)
 

### Describe your final model architecture

My final model consisted of the following layers:

|      Layer      |                 Description                 |
|:---------------:|:-------------------------------------------:|
|      Input      |              32x32x3 RGB image              |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x10 |
|      RELU       |                                             |
|   Max pooling   |        2x2 stride,  outputs 14x14x10        |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x18 |
|      RELU       |                                             |
|   Max pooling   |         2x2 stride,  outputs 5x5x10         |
| Convolution 5x5 |  1x1 stride, same padding, outputs 5x5x10   |
|      RELU       |                                             |
|     Flatten     |                 outputs 400                 |
|      RELU       |                                             |
| Fully Connected |                 outputs 240                 |
|      RELU       |                                             |
|     output      |                 outputs 43                  |

### Describe how you trained your model
I used following configure and hyperparameters in my final model:
- framework: TensorFlow
- learning rate: 0.002
- Epochs: 14
- batch size: 128
- loss operation: cross entropy
- optimiser: Adam optimiser

### Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* training set accuracy of 0.99
* validation set accuracy of 0.942
* test set accuracy of 0.92

From previous experience, I started with LeNet-5, whcih is a simple but powerful structure that is famous for image classification problems. 

After a few tries, I found that for a 43 classes case, the original amount of parameters is not big enough for the model to learn accurately. This can be seen form the phenomenon that the training accuracy is limited to around 90%. Afterward, I add another layer and more kernel to the model. This approach solved the training accuracy ceiling issue, resulting in a 97% training accuracy and 90% validation accuracy.

I tried using a dropout layer just in case that I found the model overfitting, but it was not necessary for my case and it end up setting keep_prob to 1(no drop out).

Next, I scaled down the deviation during varaible initalization to 0.05,this boost my accuracy to 99% training and 94% validating. 
Accrding to [this Stackoverflow post](https://stackoverflow.com/questions/42006089/reason-why-setting-tensorflows-variable-with-small-stddev). The reason behind this is that the small weights can be more easily effected by SGD during backprop.

After reading about the [Dying RELU](https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7), I put 0.05 to the bias term to avoid creating too many dead RELU cell. I didn't really observed the benefits from this approach but it seems convincing from the article. 

## Test a Model on New Images

### Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![](https://i.imgur.com/CuW4beU.png)

![](https://i.imgur.com/uJv343V.jpg)

![](https://i.imgur.com/d9JdOMU.jpg)

![](https://i.imgur.com/mtfkMnK.jpg)

![](https://i.imgur.com/NVTMXwi.jpg)

The last one contained watermarks and could be hard to classify.

### Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

The model predict these 5 image well, with 100% accuracy.
I guess the image I found on the internet is a little too easy for my model. However, since the test accuracy also hit 92%, averagely speaking, the 5 out of 5 correct prediction is also possible.

### Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

The top worth mentioning probabilities are listed below:

**Image 1:Roundabout**
- 100% Roundabout
- 2.5 * e-23 Speed Limit 120km/h

**Image 2:Speed Limit 60 km/h**
- 100% Speed Limit 60 km/h
- 1.4 * e-10 Speed Limit 50 km/h

**Image 3:Traffic Signal**
- 100% Traffic Signal
- 6.7 * e-15 General Caution

**Image 4:Road Work**
- 83.3% Road Work
- 15.6% Road Narrow On The Right
- 0.9%  Bicycle Crossing

**Image 5:Wild Animal Crossing**
- 99.99% Wild Animal Crossing
- 2.1 * e-4 Slippery Road
 
Note: Some probabilities that are too small to mention are neglected. The original number are listed in the end of the  jupyter notebook - [Traffic_Sign_Classifier.ipynb](https://github.com/ClarenceKuo/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb).

**Thanks for reading this!**



