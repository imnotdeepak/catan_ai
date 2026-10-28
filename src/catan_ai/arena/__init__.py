"""Four-player arena: matches, seat-rotated tournaments, replays."""

from catan_ai.arena.agents import Agent, RandomAgent
from catan_ai.arena.contestants import Contestant
from catan_ai.arena.match import MatchResult, play_match, replay, replay_record
from catan_ai.arena.tournament import rotations, run_tournament, summarize

__all__ = [
    "Agent",
    "Contestant",
    "MatchResult",
    "RandomAgent",
    "play_match",
    "replay",
    "replay_record",
    "rotations",
    "run_tournament",
    "summarize",
]
