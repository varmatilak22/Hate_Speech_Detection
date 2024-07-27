# Hate Speech Detection Project 🚫🗣️

## Overview
This project focuses on detecting hate speech in text data using natural language processing (NLP) techniques. The goal is to develop machine learning models based on decision tree and random forest algorithms to automatically identify hate speech content in text.

Project link [Hate_Speech_detection](https://hate-speech-recognition.streamlit.app/)

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Data](#data)
- [NLP Techniques](#nlp-techniques)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Getting Started
These instructions will help you set up the project on your local machine for development and testing purposes.

## Prerequisites
Make sure you have Python installed along with the necessary libraries. You can download Python from [python.org](https://www.python.org/) and install libraries using pip.

## Data
The dataset used for this project is available in the data directory. The main file is `tweets.csv`, containing text data labeled as hate speech or non-hate speech. The dataset is sourced from Kaggle.

## NLP Techniques
NLP techniques such as tokenization, stemming, and TF-IDF vectorization are utilized to preprocess the text data before model training. These techniques help in transforming raw text into numerical features that can be used by machine learning algorithms.

1. **Bag of Words (BoW)**:
   - BoW creates a list of all unique words in a text and counts how many times each word appears. It represents text as a collection of word counts.
   - For example, in the sentence "The cat sat on the mat", BoW would count "the" as 2, "cat" as 1, "sat" as 1, "on" as 1, and "mat" as 1.
2. **Tokenization**:
   - Tokenization splits text into smaller units called tokens, which can be words, phrases, or other meaningful elements.
   - For example, the sentence "I love natural language processing" would be tokenized into ["I", "love", "natural", "language", "processing"].
3. **Stemming**:
   - Stemming reduces words to their root form by removing suffixes and prefixes. This helps in grouping together words with the same meaning but different forms.
   - For example, the words "running" and "runs" would both be stemmed to "run".
4. **Count Vectorizer**:
   - Count Vectorizer converts text documents into numerical representations by counting the occurrences of words in each document.
   - It creates a matrix where each row corresponds to a document and each column corresponds to a word, with the cell values indicating the frequency of each word in each document.

## Model Training
The model training process is documented in the `model_train.py` script. Both decision tree and random forest algorithms are used to train models on the preprocessed text data.

**Decision Tree**:
   - Decision trees are a popular machine learning model for classification and regression tasks. They make predictions by recursively partitioning the feature space into smaller regions based on feature values.
   - Each node in a decision tree represents a decision based on a feature, leading to one of the possible outcomes (classes or numerical values).
   - Decision trees are easy to interpret and visualize, making them suitable for understanding the decision-making process.

## Evaluation
The performance of the trained models is evaluated using metrics such as accuracy, precision, recall, and F1-score. Detailed information is available in the evaluation section of the `model_evaluation.py` script.

## Results
The final trained models are saved in the models directory. You can use them to make predictions on new text data.

## Contributing
If you'd like to contribute to this project, please open an issue or create a pull request. All contributions are welcome! 🙌

## Acknowledgments
Kaggle
