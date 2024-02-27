# Hate Speech Detection Project 🚫🗣️
## Overview
This project focuses on detecting hate speech in text data using natural language processing (NLP) techniques. The goal is to develop machine learning models based on decision tree and random forest algorithms to automatically identify hate speech content in text.
Project link [https://colab.research.google.com/github/tilakrvarma22/installer/blob/main/Hate_Speech_Detection.ipynb]

## Table of Contents
* Getting Started
* Prerequisites
* Installation
* Usage
* Data
* NLP Techniques
* Model Training
* Evaluation
* Results
* Contributing
* License
* Acknowledgments

## Getting Started
These instructions will help you set up the project on your local machine for development and testing purposes.

## Prerequisites
Make sure you have Python installed along with the necessary libraries. You can download Python from python.org and install libraries using pip.

## Data
The dataset used for this project is available in the data directory. The main file is hate_speech_data.csv, containing text data labeled as hate speech or non-hate speech.
The dataset used was present of kaggle.

## NLP Techniques
NLP techniques such as tokenization, stemming, and TF-IDF vectorization are utilized to preprocess the text data before model training. These techniques help in transforming raw text into numerical features that can be used by machine learning algorithms.

1. Bag of Words (BoW):
  * BoW creates a list of all unique words in a text and counts how many times each word appears. It represents text as a collection of word counts.
  * For example, in the sentence "The cat sat on the mat", BoW would count "the" as 2, "cat" as 1, "sat" as 1, "on" as 1, and "mat" as 1.
2. Tokenization:
  * Tokenization splits text into smaller units called tokens, which can be words, phrases, or other meaningful elements.
  * For example, the sentence "I love natural language processing" would be tokenized into ["I", "love", "natural", "language", "processing"].
3. Stemming:
  * Stemming reduces words to their root form by removing suffixes and prefixes. This helps in grouping together words with the same meaning but different forms.
  * For example, the words "running" and "runs" would both be stemmed to "run".
4. TF-IDF (Term Frequency-Inverse Document Frequency):
  * TF-IDF measures the importance of a word in a document relative to a collection of documents. It consists of two parts:
  * Term Frequency (TF): Measures how often a word appears in a document.
  * Inverse Document Frequency (IDF): Measures how unique a word is across all documents in the collection.
  * Words with high TF-IDF scores are those that appear frequently in a document but rarely in other documents, indicating their significance to the document's content.
5. Count Vectorizer:
  * Count Vectorizer converts text documents into numerical representations by counting the occurrences of words in each document.
  * It creates a matrix where each row corresponds to a document and each column corresponds to a word, with the cell values indicating the frequency of each word in each document.
6. TfidfVectorizer:
  * TfidfVectorizer combines the TF-IDF weighting scheme with Count Vectorizer to produce a matrix of TF-IDF features.
  * It assigns higher weights to words that are more informative and discriminative for distinguishing between documents, helping in capturing the importance of words in individual documents while considering their frequency across all documents.

##  Model Training
The model training process is documented in the train_and_evaluate.py script. Both decision tree and random forest algorithms are used to train models on the preprocessed text data.
 1. Decision Tree:

    * Decision trees are a popular machine learning model for classification and regression tasks. They make predictions by recursively partitioning the feature space into smaller regions based on feature values.
    * Each node in a decision tree represents a decision based on a feature, leading to one of the possible outcomes (classes or numerical values).
    * Decision trees are easy to interpret and visualize, making them suitable for understanding the decision-making process.
2. Random Forest:

    * Random forests are an ensemble learning method that combines multiple decision trees to make predictions. Each tree in the forest is trained independently on a random subset of the training data and features.
    * During prediction, each tree in the forest outputs a prediction, and the final prediction is determined by averaging or voting over all the individual tree predictions.
    * Random forests improve upon the performance of individual decision trees by reducing overfitting and increasing robustness to noise in the data.
* Ensemble Techniques
    1. **Bagging (Bootstrap Aggregating)**:Bagging is an ensemble technique where multiple models are trained independently on different subsets of the training data, sampled with replacement (bootstrap samples).
In the context of decision trees, bagging involves training multiple decision trees on random subsets of the training data and averaging their predictions to make the final prediction.
Bagging helps reduce variance and improve the stability and accuracy of the model by reducing the impact of individual noisy samples or outliers in the training data.
    2. **Boosting:** Boosting is another ensemble technique that combines multiple weak learners (simple models) to create a strong learner.
Unlike bagging, boosting trains models sequentially, where each new model focuses on the examples that the previous models misclassified or had difficulty predicting correctly.
Boosting algorithms such as AdaBoost (Adaptive Boosting) and Gradient Boosting iteratively improve the model's performance by giving more weight to difficult examples, leading to better generalization and predictive power.
## Evaluation
The performance of the trained models is evaluated using metrics such as accuracy, precision, recall, and F1-score. Detailed information is available in the evaluation section of the train_and_evaluate.py script.

## Results
The final trained models are saved in the models directory. You can use them to make predictions on new text data.

## Contributing
If you'd like to contribute to this project, please open an issue or create a pull request. All contributions are welcome! 🙌

## License
This project is licensed under the GECCS License - see the LICENSE file for details.

## Acknowledgments
Special thanks to Dataset Source for providing the hate speech dataset.🙏

Feel free to replace placeholders like yourusername and hate-speech-detection with appropriate information for your project. Update the license file (LICENSE) accordingly.
