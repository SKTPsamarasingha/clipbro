from cilpbro import progress


def add_replay_signals(segments, heatmap):
    """Adds seg['replay'] = average replay score (0-1) over the segment, or None."""
    if not heatmap:
        for seg in segments:
            seg["replay"] = None
        progress.warn("No replay heatmap available for this video")
        return segments

    for seg in segments:
        weighted, covered = 0.0, 0.0
        for b in heatmap:
            o = min(seg["end"], b["end_time"]) - max(seg["start"], b["start_time"])
            if o > 0:
                weighted += b["value"] * o
                covered += o
        seg["replay"] = round(weighted / covered, 2) if covered else None

    progress.ok(f"Replay scores added ({len(heatmap)} heatmap buckets)")
    return segments