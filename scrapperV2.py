# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 08:48:15 2025

@author: franc
"""

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import time

API_KEY = "AIzaSyD8piZB42Xpf_xtlEVGFX7J5SZMjJ9Y2Bo"   
VIDEO_ID = "Bv4CqIxqTMA"
youtube = build("youtube", "v3", developerKey=API_KEY)

def fetch_comments(video_id, max_pages=None, sleep_between=0.1):
    comments = []
    next_page_token = None
    page_count = 0

    while True:
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            ).execute()
        except HttpError as e:
            print(f"HTTP Error: {e}")
            time.sleep(5)
            continue

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comments": snippet.get("textDisplay", "")
            })

        page_count += 1
        next_page_token = response.get("nextPageToken")
        if not next_page_token or (max_pages and page_count >= max_pages):
            break

        time.sleep(sleep_between)

    return comments

if __name__ == "__main__":
    comments = fetch_comments(VIDEO_ID)
    df = pd.DataFrame(comments)
    df.to_csv("TI18_TRUESIGHT.csv", index=False)
    print(f"Saved {len(df)} comments to youtube_comments_text_only.csv")
