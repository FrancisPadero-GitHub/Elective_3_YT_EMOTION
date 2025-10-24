from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import time
import math

API_KEY = "AIzaSyD8piZB42Xpf_xtlEVGFX7J5SZMjJ9Y2Bo"   # place your API key here
VIDEO_ID = "ceQ2XFS1tUo" # place your Video ID here in youtube
youtube = build("youtube", "v3", developerKey=API_KEY)

def fetch_all_comments(video_id, max_pages=None, fetch_replies=False, sleep_between=0.1):
    all_comments = []
    next_page_token = None
    page_count = 0

    while True:
        try:
            resp = youtube.commentThreads().list(
                part="snippet,replies" if fetch_replies else "snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            ).execute()
        except HttpError as e:
            # Simple exponential backoff on quota / transient errors
            status = getattr(e, 'status_code', None)
            print(f"HTTP error {status}: {e}")
            # backoff loop
            for attempt in range(1, 6):
                wait = 2 ** attempt
                print(f"Waiting {wait}s and retrying (attempt {attempt})...")
                time.sleep(wait)
                try:
                    resp = youtube.commentThreads().list(
                        part="snippet,replies" if fetch_replies else "snippet",
                        videoId=video_id,
                        maxResults=100,
                        pageToken=next_page_token,
                        textFormat="plainText"
                    ).execute()
                    break
                except HttpError:
                    continue
            else:
                raise

        items = resp.get("items", [])
        for item in items:
            snip = item["snippet"]["topLevelComment"]["snippet"]
            comment = {
                "commentId": item["snippet"]["topLevelComment"]["id"],
                "text": snip.get("textDisplay"),
                "author": snip.get("authorDisplayName"),
                "authorChannelId": snip.get("authorChannelId", {}).get("value"),
                "publishedAt": snip.get("publishedAt"),
                "updatedAt": snip.get("updatedAt"),
                "likeCount": snip.get("likeCount"),
                "replyCount": item["snippet"].get("totalReplyCount", 0)
            }
            # if replies were included in this response, optionally expand them
            if fetch_replies and "replies" in item:
                replies_list = []
                for r in item["replies"].get("comments", []):
                    r_s = r["snippet"]
                    replies_list.append({
                        "replyId": r.get("id"),
                        "text": r_s.get("textDisplay"),
                        "author": r_s.get("authorDisplayName"),
                        "publishedAt": r_s.get("publishedAt"),
                        "likeCount": r_s.get("likeCount")
                    })
                comment["replies_sample"] = replies_list  # note: this may be partial
            all_comments.append(comment)

        page_count += 1
        if max_pages and page_count >= max_pages:
            break

        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break

        # be polite to the API (and avoid hitting QPS limits)
        time.sleep(sleep_between)

    return all_comments

if __name__ == "__main__":
    comments = fetch_all_comments(VIDEO_ID, fetch_replies=False)
    df = pd.DataFrame(comments)
    df.to_csv("generated.csv", index=False)
    print(f"Saved {len(df)} comments to youtube_comments.csv")
