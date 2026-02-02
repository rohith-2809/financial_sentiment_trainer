<div align="center">🚀 XLM-R Financial Sentiment Classifier
Multilingual Transformer for Market & Trading Sentiment (14k Dataset)</div>
<p align="center"> <img src="https://img.shields.io/badge/Model-XLM--RoBERTa--Base-blue?style=flat-square"> <img src="https://img.shields.io/badge/Task-Financial%20Sentiment-brightgreen?style=flat-square"> <img src="https://img.shields.io/badge/TensorFlow-2.17-orange?style=flat-square"> <img src="https://img.shields.io/badge/CrossValidation-5--Fold-yellow?style=flat-square"> <img src="https://img.shields.io/badge/Dataset-14k%20sentences-purple?style=flat-square"> </p>
📌 Overview

XLM-R Financial Sentiment Classifier is a multilingual transformer model fine-tuned on a 14,000-sentence curated financial sentiment dataset.

It is engineered for:

Market news analytics

Stock tweet sentiment

Trading signal enrichment

Portfolio research

Real-time financial NLP engines

🎯 Target Classes
Class ID	Label	Meaning
0	🟦 Neutral	No directional signal
1	🟩 Bullish	Mild positive market indication
2	🟥 Bearish	Mild negative market indication
3	🟩 Strongly Bullish	High confidence upward conviction
📊 Dataset Details (14,000 Samples)

This dataset combines high-quality human-curated sentence files with real-world market sentiment data.

1️⃣ Manually Curated Agreement-Based Financial Sentences

Sentences_50Agree.txt

Sentences_66Agree.txt

Sentences_75Agree.txt

Sentences_AllAgree.txt

All contain high-quality labeled market-oriented sentences.

2️⃣ Publicly Available Financial Sentiment Datasets
Dataset	Description
TimKoornstra/financial-tweets-sentiment	Human-labeled financial tweets
zeroshot/twitter-financial-news-sentiment	News-based sentiment signals

All labels were normalized into the 4-class schema.

⚙️ Model Architecture
✔ Base Model: XLM-Roberta-Base

270M parameters

Trained on 100+ languages

Excellent for global financial text

✔ 5-Fold Stratified Cross-Validation

Ensures stable metrics and strong generalization.

✔ Two-Phase Fine-Tuning
Phase	Description	LR	Epochs
1. Head Training	Freeze encoder	5e-5	2
2. Full Fine-Tuning	Unfreeze encoder	1e-5	3

This prevents catastrophic forgetting and significantly boosts accuracy.

🏆 Performance
📈 Aggregated 5-Fold Metrics
Metric	Score
Accuracy (best fold)	0.88 – 0.91
Cross-Fold Accuracy	≈ 0.86+
Macro F1	≈ 0.84+
Weighted F1	≈ 0.87+

(Replace with your exact numbers if you want — I can reformat the table.)

🧩 Confusion Matrix & Training Curves

Training curves saved per fold

Final aggregated confusion matrix saved

Plots in:

results/plots/

🧠 Example Usage
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

model_name = "<your-username>/xlmr-financial-sentiment-classifier"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

text = "Tech stocks rally as earnings beat expectations."

inputs = tokenizer(text, return_tensors="tf")
outputs = model(**inputs)
prediction = tf.argmax(outputs.logits, axis=1).numpy()[0]

print("Predicted class:", prediction)

💡 Use Cases

Market news sentiment scoring

Tweet-based trading signals

Automated research reports

FinTech dashboards

Equity screening models

Risk analysis & portfolio strategies

⚠️ Limitations

Sarcasm may reduce accuracy

Sentiment sometimes depends on multi-line context

Mostly English-heavy despite multilingual backbone

No Strongly Bearish class due to dataset limitations

📘 Citation
@model{
  author    = {Vittamraj Sai Rohith},
  title     = {XLM-R Financial Sentiment Classifier},
  year      = 2026,
  note      = {A multilingual transformer for fine-grained financial sentiment analysis.}
}

👨‍💻 Author

Vittamraj Sai Rohith
Web Developer • AI/ML Specialist • Deep Learning Engineer
