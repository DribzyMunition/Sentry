#!/usr/bin/env python3
import json, hashlib, os, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import yaml
import tldextract
from dateutil import parser as dtparse

# --- SMART SCORING (optional) ---
import os, numpy as np
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENTRY_SMART   = os.getenv("SENTRY_SMART", "1") == "1"
EMBED_MODEL    = os.getenv("SENTRY_EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL      = os.getenv("SENTRY_LLM_MODEL", "gpt-4o-mini")
EMBED_TOP_N    = int(os.getenv("SENTRY_EMBED_TOP_N", "120"))  # cap embed calls

def _embed(text: str):
    if not (SENTRY_SMART and OPENAI_API_KEY and OpenAI):
        return None
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        r = client.embeddings.create(model=EMBED_MODEL, input=text[:4000])
        return np.array(r.data[0].embedding, dtype=np.float32)
    except Exception:
        return None

_ANCHOR = None
def _anchor_vec():
    global _ANCHOR
    if _ANCHOR is not None:
        return _ANCHOR
    seed = " ; ".join([
        " ".join(LEX.get("kinetic_keywords", [])),
        " ".join(LEX.get("casualty_keywords", [])),
        " ".join(LEX.get("commerce_disruption_keywords", [])),
        " ".join(LEX.get("regions", []))
    ])
    _ANCHOR = _embed("Conflict signals: " + seed)
    return _ANCHOR

def _embed_pts(text: str) -> int:
    a, v = _anchor_vec(), _embed(text)
    if a is None or v is None:
        return 0
    cos = float(np.dot(a, v) / (np.linalg.norm(a) * np.linalg.norm(v) + 1e-9))
    if cos >= 0.92: return 4
    if cos >= 0.88: return 3
    if cos >= 0.84: return 2
    if cos >= 0.80: return 1
    return 0

def _llm_nudge(text: str, base: int) -> int:
    if not (SENTRY_SMART and OPENAI_API_KEY and 5 <= base <= 7 and OpenAI):
        return base
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        schema = {
          "name": "SentryScore",
          "strict": True,
          "schema": {
            "type": "object",
            "properties": {"sentry_score": {"type":"integer","minimum":0,"maximum":10}},
            "required": ["sentry_score"]
          }
        }
        r = client.responses.create(
          model=LLM_MODEL,
          input=[{"role":"user","content":"Return JSON {sentry_score:0..10} only.\n\n"+text[:4000]}],
          response_format={"type":"json_schema","json_schema":schema}
        )
        js = getattr(r, "output_parsed", None) or {}
        return max(0, min(10, int(js.get("sentry_score", base))))
    except Exception:
        return base

def smart_score(text: str, base: int, do_embed: bool) -> (int, str):
    s, tag = base, ""
    if do_embed:
        pts = _embed_pts(text)
        if pts:
            s += pts
            tag += f"; embed+{pts}"
    if 5 <= s <= 7:
        s2 = _llm_nudge(text, s)
        if s2 != s:
            s = s2
            tag += "; llm"
    return max(0, min(10, int(s))), tag


ROOT = Path(__file__).parent
CFG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FEEDS_FILE = CFG_DIR / "feeds.yaml"
LEX_FILE   = CFG_DIR / "lexicon.yaml"

def now_utc():
    return datetime.now(timezone.utc)

def load_yaml(p: Path, key=None, default=None):
    if not p.exists():
        return default if key is None else default
    obj = yaml.safe_load(p.read_text()) or {}
    return obj if key is None else obj.get(key, default)

def sha_id(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p or "").encode("utf-8"))
    return h.hexdigest()[:16]

def domain_for(url: str) -> str:
    try:
        ext = tldextract.extract(url)
        return ".".join([x for x in [ext.domain, ext.suffix] if x])
    except Exception:
        return ""

def parse_time(s: str):
    try:
        return dtparse.parse(s).astimezone(timezone.utc)
    except Exception:
        return now_utc()

def pull_feed(url: str):
    d = feedparser.parse(url)
    out = []
    for e in d.entries:
        title = e.get("title", "").strip()
        link  = e.get("link", "")
        summ  = (e.get("summary", "") or e.get("description", "") or "").strip()
        pub   = e.get("published", "") or e.get("updated", "")
        ts    = parse_time(pub)
        dom   = domain_for(link or url)
        eid   = sha_id(title, link, ts.isoformat())
        out.append({
            "id": eid,
            "title": title,
            "link": link,
            "summary": summ,
            "source_domain": dom,
            "published": ts.isoformat()
        })
    return out

def score_event(ev, rules):
    text = f"{ev['title']} {ev['summary']}".lower()

    def count_hits(terms):
        n = 0
        for t in terms:
            if t.lower() in text:
                n += 1
        return n

    # 0–4: relevance via kinetic + casualty + commerce signals
    kin = min(count_hits(rules["kinetic_keywords"]), 3)
    cas = min(count_hits(rules["casualty_keywords"]), 2)
    com = min(count_hits(rules["commerce_disruption_keywords"]), 2)
    relevance = min(4, kin + (1 if kin >= 1 else 0) + (1 if cas >= 1 else 0) + (1 if com >= 1 else 0))

    # 0–2: source credibility tier
    dom = (ev.get("source_domain") or "").lower()
    tier = 0
    for d in rules.get("source_tiers", {}).get("tier3", []):
        if d in dom: tier = max(tier, 2)
    for d in rules.get("source_tiers", {}).get("tier2", []):
        if d in dom: tier = max(tier, 1)
    # tier1 explicitly 0 weight (blogs, etc.)

    # 0–2: region hit (weak but useful)
    region_hit = 0
    for r in rules.get("regions", []):
        if r.lower() in text:
            region_hit = 1
            break

    score = max(0, min(10, relevance + tier + region_hit))
    reasons = []
    if kin: reasons.append(f"kinetic x{kin}")
    if cas: reasons.append(f"casualties x{cas}")
    if com: reasons.append(f"commerce x{com}")
    if tier: reasons.append(f"source+t{tier}")
    if region_hit: reasons.append("region")
    return score, ", ".join(reasons) or "low-signal"

def weekly_rollup(events, days=7):
    cutoff = now_utc() - timedelta(days=days)
    recent = [e for e in events if dtparse.parse(e["published"]) >= cutoff]
    # naive region extraction
    def region_of(e):
        text = (e["title"] + " " + e["summary"]).lower()
        for r in LEX["regions"]:
            if r.lower() in text:
                return r
        return "Global/Unspecified"
    by_region = {}
    for e in recent:
        r = region_of(e)
        by_region.setdefault(r, 0)
        if e["sentry_score"] >= 7:
            by_region[r] += 1
    top = sorted([e for e in recent if e["sentry_score"] >= 7], key=lambda x: x["sentry_score"], reverse=True)[:10]
    return {
        "generated_at": int(time.time()),
        "window_days": days,
        "events_considered": len(recent),
        "promoted_count": len(top),
        "by_region_promoted": by_region,
        "top_events": [{
            "title": e["title"], "score": e["sentry_score"], "link": e["link"],
            "published": e["published"], "source_domain": e["source_domain"], "reasons": e["reasons"]
        } for e in top]
    }

# Load config
FEEDS = load_yaml(FEEDS_FILE, "feeds", default=[])
LEX   = load_yaml(LEX_FILE, None, default={
    "kinetic_keywords": [], "casualty_keywords": [], "commerce_disruption_keywords": [],
    "regions": [], "source_tiers": {}
})

def main():
    all_items = []
    for url in FEEDS:
        try:
            items = pull_feed(url)
            all_items.extend(items)
        except Exception as e:
            print(f"[warn] feed error {url}: {e}", file=sys.stderr)

    # de-dupe by (title+domain)
    seen = set()
    uniq = []
    for e in all_items:
        k = (e["title"], e["source_domain"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(e)

    # score
    scored = []
    for e in uniq:
        s, why = score_event(e, LEX)
        e["sentry_score"] = s
        e["reasons"] = why
        scored.append(e)

    # sort + write
    scored.sort(key=lambda x: x["sentry_score"], reverse=True)
    with (DATA_DIR / "events_scored.jsonl").open("w", encoding="utf-8") as f:
        for e in scored:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    weekly = weekly_rollup(scored, days=7)
    (DATA_DIR / "weekly_state.json").write_text(json.dumps(weekly, indent=2))

    # console summary
    print(f"SENTRY: {len(scored)} items, top 5:")
    for e in scored[:5]:
        print(f"  [{e['sentry_score']:02d}] {e['title']}  ({e['source_domain']})")

if __name__ == "__main__":
    main()
