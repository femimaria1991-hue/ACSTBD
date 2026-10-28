import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath("src"))

from cross_source_trust.attack_scenarios import schedule_non_overlapping_episodes

def test_non_overlapping_schedule():
    total_hours = 5261
    counts = {"s1": 40, "s2": 20, "s3": 20, "s4": 20}
    durations = {"s1": (3, 12), "s2": (6, 24), "s3": (2, 6), "s4": (3, 12)}
    episodes = schedule_non_overlapping_episodes(total_hours, counts, durations, min_separation=2, seed=42)

    assert len(episodes) == 100
    # Check non-overlapping and min separation of 2 hours
    for i in range(len(episodes) - 1):
        prev_end = episodes[i]["end"]
        next_start = episodes[i + 1]["start"]
        assert next_start - prev_end >= 2, f"Overlap between episode {i} and {i+1}: {prev_end} vs {next_start}"
