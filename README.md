# German Traffic Sign Classifier


**The goal of this project is to train a CNN model that identifies German traffic signs.**

## Summary
Architecture: LeNet-5 with modified setting

Data source: [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

Accuracy: 
- Training: 99.5%
- Validation: 94.2%
- Test: 92%

### Basic summary of the data set

Using the numpy and matplotlib libraries, I calculated and visualized the summary statistics of the traffic signs data set.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes in the data set is 43

Here is an exploratory visualization of the distribution of the data set: A histogam chart showing how the data is spread accross different categories.
![](https://i.imgur.com/j7kyCHw.png)

From the plots, we can see that the distributions are similar in all data sets, so the training is fairly balanced.


## Design and Test a Model Architecture

### Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?

All the pixels of training images are coverted to floating number range from 0~1 using:

pixcel = pixcel/ 255


This step will recenter the mean and deviation of the data to almost 0.

The X_min and X_max from training set data are used on both validation and test set for preprocess consistency.

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

### The approach

Final model results:
* training set accuracy of 0.99
* validation set accuracy of 0.942
* test set accuracy of 0.92

I started with LeNet-5, whcih is a simple but powerful structure that is famous for image classification problems. 

After a few tries, I found that for a 43 classes case, the original amount of parameters is not big enough for the model to learn accurately. This can be seen form the phenomenon that the training accuracy is limited to around 90%. Afterward, I add another layer and more kernel to the model. This approach solved the training accuracy ceiling issue, resulting in a 97% training accuracy and 90% validation accuracy.I added an additional dropout layer in case that the model experienced overfitting, but it was not necessary for my case and I end up setting keep_prob to 1(no drop out).

Next, I scaled down the deviation during varaible initalization to 0.05,this boost my accuracy to 99% training and 94% validating. 
Accrding to [this Stackoverflow post](https://stackoverflow.com/questions/42006089/reason-why-setting-tensorflows-variable-with-small-stddev). The reason behind this is that the small weights can be more easily effected by SGD during backprop.

After reading about the [Dying RELU](https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7), I put 0.05 to the bias term to avoid creating too many dead RELU cell. I didn't really observe the benefit from this approach but might be helpful with bigger model. 

## Test a Model on New Images

### Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![](https://i.imgur.com/7p1bejd.png)
![](https://i.imgur.com/lm7CfBf.jpg)
![](https://i.imgur.com/eUCTLz5.jpg)
![](https://i.imgur.com/LL9HnRF.jpg)
![](https://i.imgur.com/pV6QuER.jpg)

This model predict these 5 image well, with 100% accuracy.

**Thanks for reading this!**



