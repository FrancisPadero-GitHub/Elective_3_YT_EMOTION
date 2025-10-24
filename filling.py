# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 07:43:32 2025
Author: franc
Description: YouTube comments emotion detection with token-level truncation and batching.
"""

import re
import string
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# -------------------------------------------------------
# 1. Load dataset
# -------------------------------------------------------
df = pd.read_csv("TI18_TRUESIGHT.csv")

# -------------------------------------------------------
# 2. Clean up the 'comments' column
# -------------------------------------------------------
def clean_text(comments):
    if not isinstance(comments, str):
        return ""
    comments = comments.lower()
    comments = re.sub(r"http\S+", "", comments)
    comments = re.sub(r"@\w+", "", comments)
    comments = comments.translate(str.maketrans("", "", string.punctuation))
    comments = re.sub(r"\d+", "", comments)
    return comments.strip()

df["cleanedComments"] = df["comments"].astype(str).apply(clean_text)

# -------------------------------------------------------
# 3. Initialize the emotion model
# -------------------------------------------------------
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    truncation=True,
    padding=True,
    max_length=512,
    batch_size=16
)

# -------------------------------------------------------
# 4. Run predictions with progress
# -------------------------------------------------------
texts = df["cleanedComments"].tolist()
results = []

for i in tqdm(range(0, len(texts), 16), desc="Running emotion model"):
    batch = texts[i:i+16]
    results.extend(emotion_model(batch))

df["emotion"] = [r["label"] for r in results]

# -------------------------------------------------------
# 5. Save output
# -------------------------------------------------------
df.to_csv("TI18_TRUESIGHT_FILLED.csv", index=False)
print("✅ Emotion labels added! Saved as youtube_comments_with_emotion.csv")
print(df[["comments", "emotion"]].head())
