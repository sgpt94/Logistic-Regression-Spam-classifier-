# Logistic-Regression-Spam-classifier
### Building a spam e-mail filter using regularised Logistic Regression.
#### Platform : MATLAB


Please refer to the "Main.m" file
___

**Objective** : Training a classifier to classify whether a given email, x, is spam (y = 1) or non-spam (y = 0).
___
#### 1. Preprocessing Emails

Before starting on a machine learning task, it is usually insightful to take a look at examples from the dataset.
The Emails for training and testing the algorithms are downloaded from http://spamassassin.apache.org/old/publiccorpus/

In 'processEmail.m', we have implemented the following email preprocessing and normalization steps:
* **Lower-casing**
* **Stripping HTML**
* **Normalizing URLS:** All URLs are replaced with the text "httpaddr".
* **Normalizing Email Addresses:** All email addresses are replaced with the text "emailaddr"
* **Normalizing Numbers:** All numbers are replaced with the text "number".
* **Normalizing Dollars:** All dollar signs ($) are replaced with the text "dollar".
* **Word Stemming:** Words are reduced to their stemmed form. For example, "discount", "discounts", "discounted" and
"discounting" are all replaced with "discount". Sometimes, the Stemmer actually strips off additional characters from
the end, so "include", "includes", "included", and "including" are all replaced with "includ".
* **Removal of non-words:** Non-words and punctuation have been removed. All white spaces (tabs, newlines, spaces) 
have all been trimmed to a single space character.

#### 2. Vocabulary List

After preprocessing the emails, we have a list of words for each email. The next step is to choose which words we would 
like to use in our classifier and which we would want to leave out.For this project, I have chosen only the most 
frequently occuring words as our set of words considered (the vocabulary list). Since words that occur rarely in the 
training set are only in a few emails, they might cause the model to overfit our training set. The complete vocabulary 
list is in the file **'vocab.txt'**. Our vocabulary list was selected by choosing all words which occur at least a 100 
times in the spam corpus, resulting in a list of 1899 words.
Given the vocabulary list, we can now map each word in the preprocessed emails into a list of word indices that contains
the index of the word in the vocabulary list.


#### 3. Extracting features from Emails

**"emailFeatures.m"** is the script which converts the preprocessed email into a n-dimensional vector of 0s and 1s (here n = 1899).
n = 1899 since there are 1899 words in our dictionary "vocab.txt".<br>
xi = 1 indicates that the ith word in the dictionary is present in the preprocessed email.<br>
xi = 0 means the ith word is not in the email.

#### 4. Training and Testing Logictic Regression
Data is loaded from **"spamTrain.mat"** file which contains 4000 rows of preprocessed data.<br>
The regularisation parameter(lambda) is taken as 1.<br>
**"fminunc"** function is used for optimization of cost function<br>
**Training Accuracy:**  99.9% <br>
**Test Accuracy:**  99.1%
