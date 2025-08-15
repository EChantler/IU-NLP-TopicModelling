#!/usr/bin/env python3
import os
import sys
import time
import json
import yaml
import argparse
import random
from pathlib import Path
from datetime import datetime
from dateutil import tz
from typing import Dict, Any, Iterable, List, Optional, Set

import praw
from prawcore.exceptions import RequestException, ResponseException, PrawcoreException
from tqdm import tqdm

# -------------------------
# Auth
# -------------------------
def get_reddit() -> praw.Reddit:
    required = ["REDDIT_CLIENT_ID","REDDIT_CLIENT_SECRET","REDDIT_USERNAME","REDDIT_PASSWORD","REDDIT_USER_AGENT"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        username=os.environ["REDDIT_USERNAME"],
        password=os.environ["REDDIT_PASSWORD"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
        ratelimit_seconds=10,
    )

# -------------------------
# Helpers
# -------------------------
def to_iso_utc(ts: float) -> str:
    return datetime.utcfromtimestamp(ts).replace(tzinfo=tz.UTC).isoformat()

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def backoff(seconds=5):
    time.sleep(max(3, seconds))

def load_topic_map(path: Optional[str]):
    if not path:
        return None, {}
    with open(path, "r", encoding="utf-8") as f:
        m = yaml.safe_load(f)
    inv = {}
    for topic, subs in m.items():
        for s in subs or []:
            inv[str(s).lower()] = topic
    return m, inv

def serialize_submission(s, include_title=True, full=False):
    title = s.title or ""
    body = s.selftext or ""
    text = (title + "\n\n" + body).strip() if include_title else (body or title)
    base = {"text": text}
    if full:
        base.update({
            "id": s.id,
            "subreddit": str(s.subreddit),
            "flair": s.link_flair_text,
            "created_utc": to_iso_utc(s.created_utc),
            "permalink": f"https://www.reddit.com{s.permalink}",
            "is_self": bool(getattr(s, "is_self", False)),
            "over_18": bool(getattr(s, "over_18", False)),
            "num_comments": int(getattr(s, "num_comments", 0)),
            "score": int(getattr(s, "score", 0)),
            "upvote_ratio": float(getattr(s, "upvote_ratio", 0.0)) if getattr(s, "upvote_ratio", None) is not None else None,
        })
    return base

def assign_label(subreddit: str, flair: Optional[str], mode: str, inv_topic_map: Dict[str, str]) -> str:
    sr = (subreddit or "").lower()
    bucket = inv_topic_map.get(sr)
    flair_clean = (flair or "").strip()
    if mode == "subreddit":
        return bucket if bucket else subreddit
    if mode == "flair":
        return flair_clean or (bucket if bucket else subreddit)
    # hybrid default
    return flair_clean or (bucket if bucket else subreddit)

# -------------------------
# Fetching
# -------------------------
def fetch_submissions(reddit, subreddit: str, mode: str, limit: int, time_filter: str, query: Optional[str]):
    sr = reddit.subreddit(subreddit)
    if mode == "new":
        gen = sr.new(limit=limit)
    elif mode == "hot":
        gen = sr.hot(limit=limit)
    elif mode == "top":
        gen = sr.top(time_filter=time_filter, limit=limit)
    elif mode == "search":
        if not query:
            raise ValueError("search mode requires --query")
        gen = sr.search(query=query, time_filter=time_filter, sort="new", limit=limit)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    try:
        for s in gen:
            yield s
    except (RequestException, ResponseException, PrawcoreException):
        backoff(10)
        raise

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Collect Reddit posts labeled by subreddit bucket and/or flair.")
    ap.add_argument("--out", default="data/reddit_labeled.jsonl", help="Output JSONL file.")
    ap.add_argument("--mode", default="top", choices=["new","hot","top","search"], help="Listing/search mode.")
    ap.add_argument("--time-filter", default="year", choices=["all","year","month","week","day","hour"], help="For top/search.")
    ap.add_argument("--query", default=None, help="Search query (only for mode=search).")

    # Labeling
    ap.add_argument("--topic-map", default=None, help="YAML mapping Topic -> [subreddits]. Enables bucketed labeling + balancing.")
    ap.add_argument("--subreddits", default=None, help="Comma-separated subreddits if no topic map is provided.")
    ap.add_argument("--label-mode", default="hybrid", choices=["subreddit","flair","hybrid"], help="Label strategy.")
    ap.add_argument("--keep-title", action="store_true", help="Include title in text (recommended).")
    ap.add_argument("--full", action="store_true", help="Keep extra metadata fields in output.")

    # Sizing & balancing
    ap.add_argument("--target-count", type=int, default=15000, help="Total posts to collect.")
    ap.add_argument("--per-subreddit-limit", type=int, default=2000, help="Hard cap pulled per subreddit call.")
    ap.add_argument("--per-topic-limit", type=int, default=None, help="Cap per topic (used with --topic-map).")

    args = ap.parse_args()

    reddit = get_reddit()
    out_path = Path(args.out)
    seen_ids: Set[str] = set()
    total_written = 0

    topic_map, inv_topic_map = load_topic_map(args.topic_map)

    def write_rows(rows: List[Dict[str, Any]]):
        nonlocal total_written
        write_jsonl(out_path, rows)
        total_written += len(rows)

    print("== Reddit labeled collection ==")
    print(f"Mode={args.mode}, time_filter={args.time_filter}, target={args.target_count}, label_mode={args.label_mode}")
    if topic_map:
        print(f"Topics: {list(topic_map.keys())}")
        # Distribute target across topics
        topics = list(topic_map.keys())
        n_topics = max(1, len(topics))
        per_topic_target = args.per_topic_limit or max(1, args.target_count // n_topics)

        for topic in topics:
            if total_written >= args.target_count:
                break
            rows_topic: List[Dict[str, Any]] = []
            subs = topic_map.get(topic, [])
            random.shuffle(subs)  # vary order
            with tqdm(desc=f"Topic {topic}", total=per_topic_target) as pbar:
                for sr in subs:
                    if len(rows_topic) >= per_topic_target or total_written >= args.target_count:
                        break
                    try:
                        pulled = 0
                        for s in fetch_submissions(reddit, sr, args.mode, args.per_subreddit_limit, args.time_filter, args.query):
                            if total_written >= args.target_count or len(rows_topic) >= per_topic_target:
                                break
                            # minimal filtering: require some text
                            title = s.title or ""
                            body = s.selftext or ""
                            if not (title.strip() or body.strip()):
                                continue
                            if s.id in seen_ids:
                                continue
                            seen_ids.add(s.id)

                            label = assign_label(str(s.subreddit), s.link_flair_text, args.label_mode, inv_topic_map)
                            ser = serialize_submission(s, include_title=args.keep_title, full=args.full)
                            ser["label"] = label
                            rows_topic.append(ser)
                            pbar.update(1)
                            pulled += 1
                        # polite pause per subreddit
                        backoff(2)
                    except Exception as e:
                        print(f"[WARN] r/{sr}: {e}")
                        backoff(5)
                # shuffle & trim just in case
                random.shuffle(rows_topic)
                rows_topic = rows_topic[:per_topic_target]
                write_rows(rows_topic)
    else:
        if not args.subreddits:
            print("ERROR: Provide --topic-map or --subreddits")
            sys.exit(1)
        subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()]
        # Spread roughly evenly across given subs
        per_sub_target = max(1, args.target_count // max(1, len(subreddits)))
        with tqdm(total=args.target_count, desc="Collecting") as pbar:
            for sr in subreddits:
                if total_written >= args.target_count:
                    break
                batch: List[Dict[str, Any]] = []
                try:
                    for s in fetch_submissions(reddit, sr, args.mode, args.per_subreddit_limit, args.time_filter, args.query):
                        if total_written + len(batch) >= args.target_count or len(batch) >= per_sub_target:
                            break
                        title = s.title or ""
                        body = s.selftext or ""
                        if not (title.strip() or body.strip()):
                            continue
                        if s.id in seen_ids:
                            continue
                        seen_ids.add(s.id)
                        label = assign_label(str(s.subreddit), s.link_flair_text, args.label_mode, {})
                        ser = serialize_submission(s, include_title=args.keep_title, full=args.full)
                        ser["label"] = label
                        batch.append(ser)
                    random.shuffle(batch)
                    write_rows(batch)
                    pbar.update(len(batch))
                    backoff(2)
                except Exception as e:
                    print(f"[WARN] r/{sr}: {e}")
                    backoff(5)

    print(f"Done. Wrote {total_written} rows to {out_path}")

if __name__ == "__main__":
    main()
