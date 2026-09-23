# Amazon Product Review Sentiment Analysis

An end-to-end **NLP sentiment analysis project** that classifies Amazon product reviews as positive or negative using multiple NLP and deep learning approaches.

The project compares **TF-IDF, Word2Vec + LSTM, and Fine-tuned BERT** models. Based on the evaluation results, the **fine-tuned BERT model achieved the best performance and was selected for the final deployment on AWS**.

## 🚀 Project Highlights

* Sentiment classification of Amazon product reviews
* Comparison of traditional ML, LSTM, and Transformer-based approaches
* Fine-tuned **BERT (`bert-base-uncased`)** for sentiment classification
* Interactive **Streamlit** web application
* Final BERT model deployed on **AWS**
* Real-time sentiment prediction from user reviews

## 🧠 Model Comparison

| Model               | Approach               | Test Accuracy |
| ------------------- | ---------------------- | ------------: |
| TF-IDF              | TF-IDF + ML            |        ~85.4% |
| Word2Vec + LSTM     | Word Embeddings + LSTM |        ~85.6% |
| **Fine-tuned BERT** | Transformer            |      **~90%** |

Since BERT achieved the highest test accuracy, it was selected for the final production deployment.

## 🔄 Workflow

```text
Amazon Review
     ↓
Text Preprocessing
     ↓
BERT Tokenization
     ↓
Fine-tuned BERT Model
     ↓
Sentiment Prediction
     ↓
Streamlit Application
     ↓
AWS Deployment
```

## ⚙️ BERT Configuration

* Model: `bert-base-uncased`
* Maximum sequence length: 128
* Batch size: 32
* Learning rate: `2e-5`
* Epochs: 10

## 🛠️ Technologies

**Python • NLTK • spaCy • Scikit-learn • Gensim • TensorFlow • PyTorch • Transformers • BERT • LSTM • Streamlit • AWS • Git/GitHub**

## 🌐 AWS Deployment

The final application uses the **fine-tuned BERT model** for sentiment prediction and is deployed on AWS.

**Live Application:**
http://13.51.205.172:8501/

## 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/arun2544/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## 📊 Example

**Input:**

```text
The product quality is excellent and I am very happy with my purchase.
```

**Prediction:**

```text
Positive
```

## 🎯 Key Learning

This project demonstrates an end-to-end ML workflow:

**Data Processing → NLP → Model Comparison → Fine-tuning → Model Selection → Streamlit Application → AWS Deployment**

## 👨‍💻 Author

**Arun Kumar**
B.Tech, Aerospace Engineering — IIT Kanpur

GitHub: https://github.com/arun2544


