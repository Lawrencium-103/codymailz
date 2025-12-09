import json
import os

STATS_FILE = "stats.json"
DEFAULT_STATS = {"used": 7, "likes": 4}

def get_stats():
    if not os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "w") as f:
                json.dump(DEFAULT_STATS, f)
        except:
            return DEFAULT_STATS
        return DEFAULT_STATS
    
    try:
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    except:
        return DEFAULT_STATS

def increment_usage():
    data = get_stats()
    data["used"] += 1
    _save_stats(data)
    return data["used"]

def increment_likes():
    data = get_stats()
    data["likes"] += 1
    _save_stats(data)
    return data["likes"]

def _save_stats(data):
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(data, f)
    except:
        pass