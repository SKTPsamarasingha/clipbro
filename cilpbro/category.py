YT_MAP = {
    "Film & Animation": "movie_scene",
    "Gaming": "gaming",
    "Education": "tutorial",
    "Howto & Style": "tutorial",
    "Comedy": "comedy",
    "News & Politics": "news_politics",
    "Sports": "sports",
    "Music": "music",
    "Science & Technology": "tech_explainer",
    "People & Blogs": "vlog_talk",
    "Entertainment": "entertainment",
}

KEYWORDS = {
    "movie_scene": ["movie clip", "official clip", "scene", "trailer", "film", "movie"],
    "podcast_interview": ["podcast", "interview", "episode", "conversation", "q&a"],
    "debate": ["debate", "vs", "argues", "heated", "hearing", "court"],
    "comedy": ["stand-up", "standup", "funny", "comedy", "sketch", "roast"],
    "tutorial": ["how to", "tutorial", "guide", "learn", "tips", "explained"],
    "gaming": ["gameplay", "walkthrough", "let's play", "speedrun"],
    "news_politics": ["breaking", "news", "election", "senate", "press conference"],
    "sports": ["highlights", "match", "goal", "nba", "nfl", "ufc"],
}

# What makes a clip work in each niche. Goes into the payload for the LLM.
CLIP_HINTS = {
    "movie_scene": "Look for dramatic reveals, standout action beats, memorable one-liners and emotional peaks.",
    "podcast_interview": "Look for strong opinions, surprising claims, stories with a payoff and quotable lines.",
    "debate": "Look for direct clashes, interruptions, rebuttals and decisive moments.",
    "comedy": "Look for setup and punchline pairs. Include the full setup and the audience reaction.",
    "tutorial": "Look for a single complete tip with a clear result. Avoid cutting mid-step.",
    "gaming": "Look for clutch plays, big reactions and unexpected events.",
    "news_politics": "Look for key statements, confrontations and newsworthy announcements.",
    "sports": "Look for decisive plays and crowd or commentator peaks.",
    "tech_explainer": "Look for a clear self-contained explanation or a surprising insight.",
    "vlog_talk": "Look for personal stories, emotional moments and strong opinions.",
    "entertainment": "Look for high-energy, surprising or funny moments.",
    "general": "Look for self-contained moments with a clear beginning and end.",
}


def detect_category(meta, segments, override=None):
    if override:
        return override

    text = " ".join([meta.get("title", ""), meta.get("context", ""), " ".join(meta.get("tags", []))]).lower()

    # 1. Keywords in the title / description / tags
    scores = {c: sum(text.count(k) for k in kws) for c, kws in KEYWORDS.items()}
    best, hits = max(scores.items(), key=lambda kv: kv[1])
    if hits > 0:
        return best

    # 2. YouTube's own category
    for c in meta.get("categories", []):
        if c in YT_MAP:
            return YT_MAP[c]

    # 3. Transcript shape: several speakers talking quickly is probably conversation
    speakers = {s.get("speaker") for s in segments if s.get("speaker")}
    if len(speakers) >= 2:
        return "podcast_interview"
    return "general"