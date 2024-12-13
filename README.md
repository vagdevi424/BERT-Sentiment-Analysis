# Sentiment Analysis of Yelp Reviews using BERT

## Project Overview

This project focuses on leveraging the advanced transformer model **BERT (Bidirectional Encoder Representations from Transformers)** for sentiment analysis of Yelp reviews. By fine-tuning the BERT model on a dataset of customer reviews, the aim is to achieve improved accuracy in sentiment classification compared to traditional machine learning approaches such as Decision Trees regression or Naive Bayes.

## Key Features

- **Advanced NLP Techniques**: Utilizes BERT, a state-of-the-art transformer model, for sentiment analysis.
- **Context-Aware Tokenization**: Preserves the contextual relationships in reviews for improved classification.
- **Optimized Fine-Tuning**: Implements optimization techniques to enhance the model's performance while avoiding overfitting.
- **Comprehensive Evaluation**: Employs metrics like accuracy, F1 score, and AUROC for robust performance evaluation.

## Steps Involved

### 1. Data Collection and Preparation
- Collect a balanced dataset of Yelp reviews, ensuring equal representation of positive and negative sentiments.
- Preprocess the dataset by cleaning text, removing unnecessary characters, and standardizing formats.

### 2. Tokenization
- Tokenize reviews using BERT's tokenizer to preserve contextual relationships.
- Convert tokenized text into input IDs and attention masks suitable for the BERT model.

### 3. Fine-Tuning the BERT Model
- Load the pre-trained BERT model and add a classification layer.
- Fine-tune the model using optimization techniques like learning rate scheduling and dropout to avoid overfitting.

### 4. Performance Evaluation
- Evaluate the fine-tuned model using metrics like:
  - **Accuracy**: Overall correctness of predictions.
  - **F1 Score**: Balance between precision and recall.
  - **AUROC**: Ability to distinguish between positive and negative classes.

## Installation and Usage

### Prerequisites
- Python 3.8 or later
- PyTorch
- Transformers library by Hugging Face
- Scikit-learn
- Pandas, NumPy, and Matplotlib

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-yelp-bert.git
   cd sentiment-analysis-yelp-bert
