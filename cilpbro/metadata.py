import re
import yt_dlp

# Capitalized words that are usually not names
COMMON = {
    "The", "This", "That", "These", "Those", "And", "But", "For", "With", "From",
    "Subscribe", "Follow", "Like", "Watch", "Video", "Official", "Channel",
    "Instagram", "Twitter", "Facebook", "Youtube", "Click", "Link", "Music",
    "Credit", "Credits", "Copyright", "All", "Rights", "Reserved", "Part",
}


def fetch_video_context(url, max_terms=30):
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title", "") or ""
    description = (info.get("description", "") or "")[:1500]  # ignore link/credit spam at the end
    tags = info.get("tags") or []

    # Words capitalized in the middle of a sentence are likely names / terms
    text = f"{title}. {description}"
    candidates = []
    for sentence in re.split(r"[.!?\n]+", text):
        words = re.findall(r"[A-Za-z][A-Za-z'\-]+", sentence)
        for w in words[1:]:                       # skip the sentence's first word
            if w[0].isupper() and w not in COMMON and len(w) > 2:
                candidates.append(w)

    vocabulary = []
    for term in [*candidates, *tags]:
        if term.lower() not in {v.lower() for v in vocabulary}:
            vocabulary.append(term)
    vocabulary = vocabulary[:max_terms]

    first_sentence = re.split(r"(?<=[.!?])\s", description.strip())[0] if description else ""
    context = f"{title}. {first_sentence}".strip()[:300]

    return {
        "title": title,
        "context": context,
        "vocabulary": vocabulary,
        "categories": info.get("categories") or [],  # NEW: YouTube's own category
        "tags": tags,  # NEW
        "heatmap": info.get("heatmap"),  # NEW: None if YouTube has no data
    }