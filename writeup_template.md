# **Traffic Sign Recognition** 

[//]: # (Image References)

[image1]: ./Used_Images/Distribution_Training.png "Distribution"
[image2]: ./Used_Images/Distribution_Training_Augmentation.png "Distribution_Augmentation"
[image3]: ./Used_Images/Sign.png "Traffic_sign"

---


### Writeup / README

#### In this project deep learning is uesed to recognize the 43 german traffic signs. LeNet5 architecture is the basic structurer of the model, then pre-processing, for example, grayscale and normalization, is applied. Since the distribution of 43 classes are not balance, augmentation of images are added to improve the validation accuracy. After trying to change the parameters, for example, epochs, batch number, dropout rate, the validation accuracy reached 96.3%.


### Data Set Summary & Exploration

#### 1. There are three groups of data set: training set, validation set and test set.

The summary statistics of the traffic
signs data set are as this:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how each class of traffic sign data distributed. It is shown that, some classes have much more data than the other classes. The maximal number of one class is 
![alt text][image1]
In the following, for each sign class one example is shown.
![alt text][image3]
### Design and Test a Model Architecture

#### 1. Pre-Proccesing of the data set
Only grayscale and normalization are used to pre-process the images. As a first step, I decided to convert the images to grayscale because ...
Here is an example of a traffic sign image before and after grayscaling.As a last step, I normalized the image data because 

I generated additional data for the sign class, in which the total number is less than 750. After augmentation the difference between different classes are less. The characteristics of the augmented training set are as follows.
![alt text][image2]

#### 2. Final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1  image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected	(1)	| 400 inputs, 300 outputs       									|
| Fully connected	(2)	| 300 inputs, 120 outputs       									|
| dropout				| 0.8     									|
| Fully connected	(3)	| 120 inputs, 43 outputs       									|

 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an 
sigma for the random generated data has effect on the 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of  96.3% 
* test set accuracy of 93.6
Why I chose LeNet Architecture, because it is a proved good architecture for traffic sign classification. The input data of LeNet has one feature, whereas, colorful traffic sign image has three features. Therefore, the number of features should be changed. And the classified groups for LeNet are 10, whereas we need 43. Therefore, the output number should be corrected. 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected (1)		| 400 inputs, 120 outputs        									|
| Fully connected (2)		| 120 inputs, 84 outputs        									|
| Fully connected (3)		| 84 inputs, 34 outputs        									|
| Softmax				| etc.        									|
|						|												|
|						|												|


|			mu	|	sigma	|dropout|EPOCHS|BATCH_SIZE|rate|
|:----:|:-----:|:------:|:-----:|:------:|:-----:| 
|			0		|		0.1		|None|10|128|0.001|

with the initial architecture the validation accuracy has 90%. The validation accuracy with augmention is 88.6%.
Increase the EPOCHS to 50, which improve the validation accuracy to 92%. But it also shows overfitting.
Then rate from 0.001 to 0.002, the maximal validation acccuracy in 10 EPOCHS is 90%, and there is overfitting after 5 EPOCHS alredy.
sigma from 0.1 to 0.2, the validation accuracy is 90%

Try to change the input and output number in full-connected layer, 400, 320, 180, 43, the validation accuracy increased.89.5% 
Try to add dropout in each full-connected layer, , dropout 0.75 in the first fully connnected layer, 87.7%
dropout 0.75 in the second fully connnected layer, 89.2%
dropout 0.75 in the third fully connected layer, 92.4%
dropout 0.8 in the third fully connected layer, 92.5%
dropout 0.7 in the third fully connected layer, 90%

change the first layer number of features, 10, 88.6%
change the first layer number of features, 4, 89%
180-->200, 92.8% 
As the start accuracy is not low, therefore,
In preprocessing of image I have 
grayscale

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image4] 

The first, fifth and seventh image might be difficult to classify because the traffic sign is small compared with the whole image. The second and eighth are quite similar, which might different to identify. The sixth and ninth images are clear, and there is no any object in the image, they might be easy to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 14 Stop Sign      		| 20 Dangerous curve to the right   									| 
| 33 Turn right ahead     			| 30 Beware of ice/snow 										|
| 18 General caution				| 28 Yield											|
| 2 Speed limit 50 km/h	      		| 34 Turn left ahead 				|
| 4 Speed limit 70 km/h			| 40 Roundabout mandatory      							|
| 11 Right-of-way at the next intersection		| 11 Right-of-way at the next intersection  	|
| 31 Wild animals crossing		| 37 Turn right ahead  	|
| 40 Roundabout mandatory 		| 33 Right-of-way at the next intersection  	|
| 12 Priority road		| 12 Priority road	  	|
| 23 Slippery road		| 40 Roundabout mandatory   	|


The model was able to correctly guess 2 of the 10 traffic signs, which gives an accuracy of 20%. This compares favorably to the accuracy on the test set is low.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| 30 Beware of ice/snow  									| 
| .15     				| 40 Roundabout mandatory 										|
| .14					| 17 No entry											|
| .09	      			| 38 Keep right					 				|
| .02				    | 1 Speed limit 30     							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .94         			| 30 Beware of ice/snow 									| 
| .05     				| 29 Bicycle crossing 										|
| .01					| 38 Keep right											|
| .01	      			| 28 Children crossing					 				|
| .00				    | 23 Slippery Road      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| 28 Children crossing									| 
| .00     				| 41 End of no passing 										|
| .00					| 36 Go straight or right											|
| .00	      			| 9 No passing					 				|
| .00				    | 3 Speed limit 60      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .85         			| 23 Slippery road							| 
| .12     				| 34 Turn left ahead 										|
| .01					| 28 Children crossing											|
| .00	      			| 30 Beware of ice/snow					 				|
| .00				    | 37 Go straight or left      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| 40 Roundabout mandatory									| 
| .02     				| 28 Children crossing 										|
| .00					| 37 Go straight or left											|
| .00	      			| 27 Pedestrians					 				|
| .00				    | 29 Bicycle crossing        							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| 11 	Right-of-way at the next intersection								| 
| .00     				| 30 Beware of ice/snow 										|
| .01					| 40 Roundabout mandatory											|
| .00	      			| 19 Dangerous	curve to the left				 				|
| .00				    | 12 Priority Road      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .997         			| 37 Go straight or left									| 
| .00     				| 26 Traffic signals 										|
| .00					| 40 Roundabout mandatory											|
| .00	      			| 39 Keep left					 				|
| .00				    | 0 Speed limit 20      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .996         			| 33 Turn right ahead							| 
| .00     				| 12 Priority Road										|
| .00					| 14 Stop											|
| .00	      			| 35 Ahead only					 				|
| .00				    | 25 Road work      				|


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .996         			| 12 Priority Road									| 
| .00     				| 40 Roundabout mandatory 										|
| .00					| 18 General caution											|
| .00	      			| 15 No vehicles					 				|
| .00				    | 28 Children crossing      				|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .96         			| 34 Turn left ahead									| 
| .02    				| 37 Go straight or left 										|
| .01					| 14 Stop											|
| .01	      			| 1 Speed limit 30				 				|
| .00				    | 40 Roundabout mandatory      				|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


