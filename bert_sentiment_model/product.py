from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI(title="BERT Sentiment Analysis API")

MODEL_PATH = "./bert_sentiment_model"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()


# Request data format
class Review(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "BERT Sentiment Analysis API is running"
    }


@app.post("/predict")
def predict(review: Review):

    # Convert text into BERT input
    inputs = tokenizer(
        review.text,
        add_special_tokens=True,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Move input to same device as model
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probabilities = torch.softmax(outputs.logits, dim=1)

    # Get predicted class
    prediction = torch.argmax(probabilities, dim=1).item()

    # Convert class number to sentiment
    sentiment = "Positive" if prediction == 1 else "Negative"

    # Confidence
    confidence = probabilities[0][prediction].item()

    return {
        "text": review.text,
        "sentiment": sentiment,
        "confidence": round(confidence, 4)
    }