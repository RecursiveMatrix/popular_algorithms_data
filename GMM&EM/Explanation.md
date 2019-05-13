# Gaussian Mixture Model and Expectation Maximization
- Gaussian Mixture Model  
GMM is a linear combination of K Gaussian distributions used for clustering problems.  
Such model is based on the assumption that the given dataset is the mixture of multiple Gaussian distributions. The goal is to 
find the variances and means of all distributions.  

- Expectation Maximization  
Maximization of the likehood functions with updated means and variances.  
Apply a matrix W with the dimension of number of samples * number of groups, the content of a particular row indicates:
for a given nth sample, the probabilities it belongs to different groups.  
The goal of EM is to find an optimal matrix W using likehood function, then updating the means and variances of 
the distributions as well as the probabilities recursively.  
# Dataset
The toy dataset is generated mannuly with known means and variances so the result could be checked easily.  
5 Gaussian distributions have been created with pre-defined means and variances.  
# Fuctions
First of all, initializing the matix W, means and variances. 
Note: the selection of initial means and variances is crucial, unproperly selected values may totally lead to an awful result.
In real world, some pre-investigations or intuative insights should be performed and considered at first to narrow down the value
to a relative convinced range.  
  
  `logLH`: this function is used to calculate the sum of probabilities of every sample, then the mean of log of it.  
  `update_W`: this function returns the updated matrix W with the known means and variances.  
  `update_prob_cluster`: this function returns the probabilities which indicates the ratio of each group using updated W.  
  `update_AV`: update means using the dataset and W.  
  `update_VAR`: update variances using dataset,means and W.  
# Result  
