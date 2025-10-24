# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 07:43:32 2025

Author: franc
Description: YouTube comment emotion detection with token-level truncation and batching.
"""

import re
import string
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# -------------------------------------------------------
# 1. Load dataset
# -------------------------------------------------------
df = pd.read_csv("2018_dataset.csv")

# -------------------------------------------------------
# 2. Clean up the 'text' column
# -------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # remove links
    text = re.sub(r"@\w+", "", text)         # remove mentions
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)          # remove numbers
    text = text.strip()
    return text

df["cleanedComments"] = df["text"].astype(str).apply(clean_text)

# -------------------------------------------------------
# 3. Initialize the emotion classification model
# -------------------------------------------------------
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    truncation=True,        # token-level truncation
    padding=True,           # uniform batch size
    max_length=512,         # model-safe limit
    batch_size=16           # process 16 comments at once
)

# -------------------------------------------------------
# 4. Batch prediction with progress bar
# -------------------------------------------------------
tqdm.pandas(desc="Predicting emotions")

texts = df["cleanedComments"].tolist()
results = list(
    tqdm(
        emotion_model(texts, truncation=True, padding=True, max_length=512, batch_size=16),
        total=len(texts),
        desc="Running emotion model"
    )
)

# Extract the emotion labels safely
df["emotion"] = [r[0]["label"] if isinstance(r, list) and isinstance(r[0], dict) else r["label"] for r in results]

# -------------------------------------------------------
# 5. Save new dataset
# -------------------------------------------------------
df.to_csv("youtube_comments_with_emotion.csv", index=False)

print("✅ Emotion labels added! Saved as youtube_comments_with_emotion.csv")
print(df[["text", "emotion"]].head())
