from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple 
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["negative", "neutral", "positive"]

def estimate_sentiment(news) -> Tuple[float, str]:
    if not news:
        return 0.0, "neutral"

    tokens = tokenizer(news, return_tensors="pt", padding=True, truncation=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        logits = model(**tokens).logits

    # Compute probabilities for each headline
    probs = torch.nn.functional.softmax(logits, dim=1)
   

    # Average the probabilities across all headlines
    avg_probs = probs.mean(dim=0)

    # Determine sentiment
    sentiment_idx = torch.argmax(avg_probs).item()
    sentiment = labels[sentiment_idx]
    probability = avg_probs[sentiment_idx].item()



    return probability, sentiment


if __name__ == "__main__":
    probability, sentiment = estimate_sentiment(['markets responded negatively to the news!','traders were displeased!'])
    print(probability, sentiment)
    print(torch.cuda.is_available())



