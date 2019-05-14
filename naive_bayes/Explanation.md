# Prediction of sex based on names using Naive Bayes
The project is to predict the sex of a person based on his/her name.  
The training dataset is a bunch of names with the label indicating their real sex while the test dataset are only names.

# Overview
Bayes equation:  
![image](https://github.com/RecursiveMatrix/popular_algorithms_data/blob/master/naive_bayes/screenshorts/bayes_equation.jpg)

Prediction of the probability of a paticular name:  
![image](https://github.com/RecursiveMatrix/popular_algorithms_data/blob/master/naive_bayes/screenshorts/bayes_pred.jpg)

Notation:  
`A`: sex  
`B`: specific word  
The word is assumed to be independent with each other. As a result, the calculation of the probabilities can be simplified.

# Steps
- Calculate the frequency of each word appeared in two groups:
  - `total`: a dict indicates the number of samples of girls and boys;
  - `frequency_list_f` and `frequency_list_m` indicate two dicts, which store the frequency of a particular word when the sex is known;
  - `LaplaceSmooth` is a function that perform Laplace Smooth in order to solve the problem that some words may only be shown in the training dataset. The formulation could be shown as below:  
  ![image](https://github.com/RecursiveMatrix/popular_algorithms_data/blob/master/naive_bayes/screenshorts/Laplace_smooth.jpg)  
where nx denotes the number of appearence of a particular word, l denotes the length of a name, c denotes the number of distinct words in a name,α denotes the Laplace smooth coefficience.

- The probability of sex based on a particular name consists of the folloing two parts of probabilities :  
   - `base_f` and `base_m` indicates the log probability which can be shown as below:  
    ![image](https://github.com/RecursiveMatrix/popular_algorithms_data/blob/master/naive_bayes/screenshorts/base_prob.jpg)  
   - for a particular word based on certain sex, the probability of this word appears minus the probability of this word not appears;
   - Considering float error, we need further get the log value of them.
- Finally, depending on the result of prediction, label could be attached accordingly.  
