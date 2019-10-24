# Brain_one-data-scientist-case-study
A case study from Brain one for data scientist position. 

We want to find a supervised way to classify a review (Positive, Negative or Neutral) of a certain amazon
Product. Let’s choose the ‘Clothing , Shoes and Jewelry’ category for our Task. The data set can be downloaded from [here](http://jmcauley.ucsd.edu/data/amazon/). 

The data is in json format and contains following information. 

```python
{'reviewerID': 'A1KLRMWW2FWPL4',
 'asin': '0000031887',
 'reviewerName': 'Amazon Customer "cameramom"',
 'helpful': [0, 0],
 'reviewText': "This is a great tutu and at a really great price. It doesn't look cheap at all. I'm so glad I looked on Amazon and found such an affordable tutu that isn't made poorly. A++",
 'overall': 5.0,
 'summary': 'Great tutu-  not cheaply made',
 'unixReviewTime': 1297468800,
 'reviewTime': '02 12, 2011'}
 ```
 where:
 ```
reviewText: the review of the Product
overall: the rating of the review.
```
We have to respect this Rules :
```
- rating is >= 1 && <=2 ---> Negative (Label -1)
- rating is >=4 && <=5 ---> Positive(Label +1)
- rating is = 3 --------------> neutral (Ignore)
```
Which traslate in python as follows:
```python
df.loc[df.overall == 5.0, 'rating'] = "+1"
df.loc[df.overall == 4.0, 'rating'] = "+1"
df.loc[df.overall == 2.0, 'rating'] = "-1"
df.loc[df.overall == 1.0, 'rating'] = "-1"
```
>Note: you have to analyse the review of the Product using some NLP methods, not using the overall Values.
Let's see the count of overall in our dataset

![](https://github.com/nirajdevpandey/Brain_one-data-scientist-case-study/blob/master/plots/overall_count.png)

We can see that the most data points have recieved good reviews around 5.0. Now let's clean the data ans see the classes (positive and nagetive). 
![](https://github.com/nirajdevpandey/Brain_one-data-scientist-case-study/blob/master/plots/class_dist.png)

Here one can see that we have `very huge class imbalance`. The can be solved by using sampling and other techniques. However, for the sake of this implementation we are not perfroming any of these.

The exploratory data analisys can be seen in Python notebook. It contains both Word embedding part and the data-set part. 
### How to run?
Following thing needed to be set before exicuting ML modelling.  

```txt
1) Install dependencies
2) Download word embedding
3) Add dataset path to the script 
4) Add embedding path to the script
5) Exicute Brain_one.py
```
