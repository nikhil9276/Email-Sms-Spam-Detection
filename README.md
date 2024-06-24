                           Email-SMS-Spam-Detection
Project Overview
The Email-SMS-Spam-Detection project is aimed at building a machine learning model to classify messages (emails or SMS) as either spam or not spam (ham). This is achieved using natural language processing (NLP) techniques and supervised learning algorithms.

Key Components:
         Data Preprocessing: Text data from messages undergoes preprocessing steps such as tokenization, stop word removal, 
         punctuation removal, and stemming using NLTK (Natural Language Toolkit).

Feature Extraction: 
            Messages are transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) 
            vectorization, which converts text data into a matrix of TF-IDF features.

Model Training:

Multinomial Naive Bayes:
              A popular algorithm for text classification tasks, trained on the TF-IDF transformed data to predict whether a 
            message is spam or not.
Voting Classifier:
            Combines predictions from multiple algorithms (SVM, Multinomial Naive Bayes, Extra Trees Classifier) to improve 
           classification accuracy.
Stacking Classifier:
          Uses a meta-classifier (Random Forest Classifier) to combine predictions from multiple base classifiers for 
          enhanced performance.
Model Evaluation:
           Metrics such as accuracy and precision are used to evaluate the performance of each classifier on test data, 
           ensuring robustness and reliability of the model.

Usage:
Input: Users can input a message (email or SMS) into the application.
Output: The application predicts whether the message is likely spam or not spam (ham) based on the trained machine learning model.
Technologies Used:
Python
Scikit-learn
NLTK (Natural Language Toolkit)
Streamlit (for building the interactive web application)
GitHub (for version control and project hosting)

