# **Traffic Sign Recognition** 

[//]: # (Image References)

[image1]: ./Used_Images/Distribution_Training.png "Distribution"
[image2]: ./Used_Images/Distribution_Training_Augmentation.png "Distribution_Augmentation"
[image3]: ./Used_Images/Sign.png "Traffic_sign"
[image4]: ./Used_Images/Figure_Processing.png "Figure_Processing"
[image5]: ./Used_Images/Transform_Processing.png "Transform_Processing"
[image6]: ./Used_Images/Image_Online.png "Image_Online"


---

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
Only grayscale and normalization are used to pre-process the images. It shows the difference of before after pre-processing.
![alt text][image4]
In many articles, augmentation is proven to improve the prediction accuracy. (https://chatbotslife.com/german-sign-classification-using-deep-learning-neural-networks-98-8-solution-d05656bf51ad) And then I generated additional data for the sign class, in which the total number is less than 750. Three augmented images are created with rotation, scaling and transformation. Here is an example of augmented images:
![alt text][image5]
After augmentation the difference between different classes are less. The characteristics of the augmented training set are as follows.
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



#### 3. Approach to find a solution to have high accuracy and to train the model

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of  96.3% 
* test set accuracy of 93.6

LeNet architecture is the initial model I choose from course. Why I chose LeNet Architecture? Because it is a proved good architecture for traffic sign classification. The input data of LeNet has one feature, whereas, colorful traffic sign image has three features. Therefore, the number of features should be changed. And the classified groups for LeNet are 10, whereas we need 43. Therefore, the output number should be corrected. The original model consists the following layers:

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

With the initial model and with the values of parameters in the following, the validation accuracy is around 89%.

|			mu	|	sigma	|dropout|EPOCHS|BATCH_SIZE|rate|
|:----:|:-----:|:------:|:-----:|:------:|:-----:| 
|			0		|		0.1		|None|10|128|0.001|

Then I tried to increase or decrease one of the parameters once to check if the change can imorove the accuracy.
1. Increase the EPOCHS to 50, which improve the validation accuracy to 92%. But it also shows overfitting.
2. Increase iteration rate from 0.001 to 0.002, the maximal validation acccuracy in 10 EPOCHS is 90%.
sigma from 0.1 to 0.2, the validation accuracy is 90%
3. Sigma can improve the start accuracy but hasn't improved the final result.
4. BATCH_SIZE can also not improve the accuracy obvious.
5. Try to change the input and output number in full-connected layer, 400, 320, 180, 43, the validation accuracy increased to 89.5%. Try to the output number of each full-connected layer, it shows the output number of the third fully connected layer has largest effect on the accuracy. 
6. Try to add dropout in each full-connected layer seperately. With dropout 0.75 in the first fully connnected layer the accuracy is 87.7%. With dropout 0.75 in the second fully connnected layer, the result is 89.2%. With dropout 0.75 in the third fully connected layer, the accuracy increased to 92.4%, which is a big improvement. Then I tried dropout 0.85, 0.8 and 0.7 in the third fully connected layer, dropout 0.8 is the best choice.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image6] 

The first, fifth and seventh image might be difficult to classify because the traffic sign is small compared with the whole image. The second and eighth are quite similar, which might different to identify. The sixth and ninth images are clear, and there is no any object in the image, they might be easy to classify.In third, fourth and tenth there is other object, it is difficult to say, if it is easy to classify correctly.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 14 Stop Sign      		| 20 Dangerous curve to the right   									| 
| 33 Turn right ahead     			| 30 Beware of ice/snow 										|
| 18 General caution				| 28 Children crossing											|
| 2 Speed limit 50 km/h	      		| 34 Turn left ahead 				|
| 4 Speed limit 70 km/h			| 40 Roundabout mandatory      							|
| 11 Right-of-way at the next intersection		| 11 Right-of-way at the next intersection  	|
| 31 Wild animals crossing		| 37 Turn right ahead  	|
| 40 Roundabout mandatory 		| 33 Right-of-way at the next intersection  	|
| 12 Priority road		| 12 Priority road	  	|
| 23 Slippery road		| 40 Roundabout mandatory   	|


The model was able to correctly guess 2 of the 10 traffic signs, which gives an accuracy of 20%. This compares favorably to the accuracy on the test set is low.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the last second cell of the Ipython notebook.

For the first image Stop Sign, the model is relatively sure that this is a Beware of ice/snow sign (probability of 0.6). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| 30 Beware of ice/snow  									| 
| .15     				| 40 Roundabout mandatory 										|
| .14					| 17 No entry											|
| .09	      			| 38 Keep right					 				|
| .02				    | 1 Speed limit 30     							|

For the second image Turn right ahead, the model is very sure that this is a Beware of ice/snow sign (probability of 0.94). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .94         			| 30 Beware of ice/snow 									| 
| .05     				| 29 Bicycle crossing 										|
| .01					| 38 Keep right											|
| .01	      			| 28 Children crossing					 				|
| .00				    | 23 Slippery Road      							|

For the third image General caution, the model is very sure that this is a Children crossing (probability of 1). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| 28 Children crossing									| 
| .00     				| 41 End of no passing 										|
| .00					| 36 Go straight or right											|
| .00	      			| 9 No passing					 				|
| .00				    | 3 Speed limit 60      							|

For the forth image Speed limit 50 km/h, the model is very sure that this is a slippery road sign (probability of 0.85). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .85         			| 23 Slippery road							| 
| .12     				| 34 Turn left ahead 										|
| .01					| 28 Children crossing											|
| .00	      			| 30 Beware of ice/snow					 				|
| .00				    | 37 Go straight or left      							|

For the fifth image Speed limit 70 km/h, the model is very sure that this is a slippery road sign (probability of 0.98). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| 40 Roundabout mandatory									| 
| .02     				| 28 Children crossing 										|
| .00					| 37 Go straight or left											|
| .00	      			| 27 Pedestrians					 				|
| .00				    | 29 Bicycle crossing        							|

For the sixth image Right-of-way at the next intersection, the model is very sure that this is Right-of-way at the next intersection (probability of 1). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| 11 	Right-of-way at the next intersection								| 
| .00     				| 30 Beware of ice/snow 										|
| .01					| 40 Roundabout mandatory											|
| .00	      			| 19 Dangerous	curve to the left				 				|
| .00				    | 12 Priority Road      							|

For the seventh image Wild animals crossing, the model is very sure that this is Go straight or left (probability of 0.997). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .997         			| 37 Go straight or left									| 
| .00     				| 26 Traffic signals 										|
| .00					| 40 Roundabout mandatory											|
| .00	      			| 39 Keep left					 				|
| .00				    | 0 Speed limit 20      							|

For the eighth image Roundabout mandatory, the model is very sure that this is Turn right ahead (probability of 0.996). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .996         			| 33 Turn right ahead							| 
| .00     				| 12 Priority Road										|
| .00					| 14 Stop											|
| .00	      			| 35 Ahead only					 				|
| .00				    | 25 Road work      				|

For the ninth image Priority road, the model is very sure that this is Priority road (probability of 0.996). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .996         			| 12 Priority Road									| 
| .00     				| 40 Roundabout mandatory 										|
| .00					| 18 General caution											|
| .00	      			| 15 No vehicles					 				|
| .00				    | 28 Children crossing      				|

For the tenth image Slippery road, the model is very sure that this is Turn left ahead (probability of 0.96). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .96         			| 34 Turn left ahead									| 
| .02    				| 37 Go straight or left 										|
| .01					| 14 Stop											|
| .01	      			| 1 Speed limit 30				 				|
| .00				    | 40 Roundabout mandatory      				|




