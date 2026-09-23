# Amazon Product Review Sentiment Analysis

An end-to-end **Natural Language Processing (NLP) and Deep Learning project** for classifying Amazon product reviews based on sentiment. The project compares traditional machine learning, Word2Vec + LSTM, and BERT-based approaches and provides an interactive web application for making predictions.

The application has also been **deployed on AWS** and is accessible through a web interface.

## 🚀 Project Overview

Customer reviews contain valuable information about user opinions and experiences. This project analyzes Amazon product reviews and predicts their sentiment using multiple NLP techniques.

The project implements and compares:

* **TF-IDF + Machine Learning**
* **Word2Vec + LSTM**
* **BERT-based Sentiment Classification**

A Streamlit application provides an interactive interface where users can enter a product review and receive a sentiment prediction.

## ✨ Features

* Amazon product review sentiment classification
* Text preprocessing using NLP techniques
* TF-IDF based sentiment model
* Word2Vec embeddings with LSTM
* BERT-based sentiment classification
* Interactive Streamlit web application
* REST/API-based prediction workflow
* AWS deployment
* Real-time sentiment prediction from user input

## 🧠 Models Used

### 1. TF-IDF + Machine Learning

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text into numerical features.

The resulting features are then used to train a machine-learning classifier.

**Test Accuracy:** ~85.4%

### 2. Word2Vec + LSTM

Word2Vec is used to generate dense word embeddings, which are then passed to an LSTM network to capture sequential information from the review text.

**Test Accuracy:** ~85.6%

### 3. BERT

A pre-trained `bert-base-uncased` model is fine-tuned for sentiment classification.

Key training configuration:

* Model: `bert-base-uncased`
* Maximum sequence length: 128
* Batch size: 32
* Learning rate: `2e-5`
* Epochs: 10

**Test Accuracy:** ~90%

The BERT-based approach achieved the highest test accuracy among the approaches evaluated in this project.

## 🔄 NLP Pipeline

The overall workflow is:

```text
Amazon Product Reviews
        ↓
Data Cleaning
        ↓
Text Preprocessing
        ↓
Tokenization
        ↓
Feature / Embedding Generation
        ↓
┌─────────────────────────────┐
│                             │
│  TF-IDF       Word2Vec      │
│    ↓             ↓          │
│  ML Model       LSTM        │
│                             │
│          BERT               │
│           ↓                 │
│    Transformer Model        │
│                             │
└─────────────────────────────┘
        ↓
Sentiment Prediction
        ↓
Streamlit Web Application
        ↓
AWS Deployment
```

## 🛠️ Technologies Used

### Programming

* Python
* SQL

### NLP & Machine Learning

* NLTK
* spaCy
* Scikit-learn
* Gensim
* TensorFlow
* PyTorch
* Transformers
* BERT
* Word2Vec
* LSTM

### Application

* Streamlit
* REST API

### Deployment & Cloud

* AWS
* Git
* GitHub

## 📊 Model Comparison

| Model           | Technique                        | Test Accuracy |
| --------------- | -------------------------------- | ------------: |
| TF-IDF          | TF-IDF + Machine Learning        |        ~85.4% |
| Word2Vec + LSTM | Word Embeddings + LSTM           |        ~85.6% |
| BERT            | Transformer-based Classification |          ~90% |

> Accuracy values are based on the evaluation performed during the project and may vary depending on preprocessing, dataset split, and training configuration.

## 🌐 AWS Deployment

The Streamlit application has been deployed on **Amazon Web Services (AWS)** and can be accessed through a web browser.

The deployed application allows users to:

1. Enter an Amazon product review.
2. Submit the review.
3. Process the review using the trained model.
4. Receive the predicted sentiment.

### Live Application

**AWS Deployment:**
`[Add your AWS application URL here]`

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/arun2544/amazon-sentiment-analysis.git
```

```bash
cd amazon-sentiment-analysis
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```powershell
streamlit run app.py
```

The application should then be available locally at:

```text
http://localhost:8501
```

## 📁 Project Structure

```text
amazon-single/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model files
├── preprocessing files
├── dataset files
└── other project files
```

> The exact structure may vary depending on the files included in the repository.

## 📌 Example

### Input

```text
The product quality is excellent and I am very happy with my purchase.
```

### Prediction

```text
Positive
```

Another example:

### Input

```text
The product stopped working after a few days. Very disappointed.
```

### Prediction

```text
Negative
```

## 🎯 Project Objectives

The main objectives of this project are:

* Understand practical NLP preprocessing.
* Compare traditional machine-learning approaches with deep-learning models.
* Implement word embeddings using Word2Vec.
* Understand sequence modelling using LSTM.
* Fine-tune a transformer-based BERT model.
* Build an interactive sentiment-analysis application.
* Deploy an ML application on AWS.
* Understand the complete workflow from model development to deployment.

## 🔮 Future Improvements

Possible improvements include:

* Improving the BERT model through further hyperparameter tuning.
* Using larger and more diverse review datasets.
* Adding sentiment confidence scores.
* Supporting multi-class sentiment such as positive, neutral, and negative.
* Adding batch prediction for multiple reviews.
* Adding model monitoring and logging.
* Containerizing the application using Docker.
* Adding CI/CD for automated deployment.

## 👨‍💻 Author

**Arun Kumar**

B.Tech, Aerospace Engineering
IIT Kanpur

GitHub: https://github.com/arun2544

## 📄 License

This project is intended for educational and portfolio purposes.
