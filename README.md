# SENTRY — Signal-Filtered Event Relevance

Pulls a few public RSS feeds, normalizes items, and assigns a 0–10 SENTRY score using a simple rules model.
Outputs:
- data/events_scored.jsonl (one event per line)
- data/weekly_state.json (rollup of last 7 days)

## Quickstart
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python sentry.py

# Optional: edit feeds and lexicon
# feeds:   config/feeds.yaml
# rules:   config/lexicon.yaml
