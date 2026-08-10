import json
import os
import time
from collections import defaultdict


DATA_DIR = "data"
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.jsonl")

os.makedirs(DATA_DIR, exist_ok=True)


def normalize_city_key(city):
    return city.strip().lower()


def save_feedback(city, poi_id, vote):
    if vote not in {"up", "down"}:
        raise ValueError("Vote must be 'up' or 'down'.")

    event = {
        "ts": time.time(),
        "city_key": normalize_city_key(city),
        "poi_id": poi_id,
        "vote": vote
    }

    with open(
        FEEDBACK_FILE,
        "a",
        encoding="utf-8"
    ) as file:
        file.write(
            json.dumps(event) + "\n"
        )

    return event


def load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return []

    events = []

    with open(
        FEEDBACK_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                events.append(
                    json.loads(line)
                )
            except json.JSONDecodeError:
                continue

    return events


def feedback_boost_map(city):
    city_key = normalize_city_key(city)

    boosts = defaultdict(float)

    for event in load_feedback():

        if event.get("city_key") != city_key:
            continue

        poi_id = event.get("poi_id")
        vote = event.get("vote")

        if vote == "up":
            boosts[poi_id] += 0.25

        elif vote == "down":
            boosts[poi_id] -= 0.35

    return dict(boosts)


def feedback_stats(city=None):
    events = load_feedback()

    if city:
        city_key = normalize_city_key(city)

        events = [
            event
            for event in events
            if event.get("city_key") == city_key
        ]

    upvotes = sum(
        1
        for event in events
        if event.get("vote") == "up"
    )

    downvotes = sum(
        1
        for event in events
        if event.get("vote") == "down"
    )

    return {
        "total": len(events),
        "upvotes": upvotes,
        "downvotes": downvotes
    }