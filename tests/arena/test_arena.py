"""Arena: rotation, determinism, exact replay, outputs (ENG-008)."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from catan_ai.arena import (
    Contestant,
    MatchResult,
    play_match,
    replay,
    replay_record,
    rotations,
    run_tournament,
)
from catan_ai.arena.contestants import AGENTS
from catan_ai.cli.arena import main as arena_main
from catan_ai.engine import ActionKind, CanonicalAction
from catan_ai.observations import PlayerObservation

LINEUP = [
    Contestant("random"),
    Contestant("catanatron:W"),
    Contestant("catanatron:F"),
    Contestant("random"),
]


def timeless(result: MatchResult) -> MatchResult:
    return replace(result, wall_seconds=0.0, decision_seconds=(0.0,) * 4)


def test_rotations_put_every_contestant_in_every_seat() -> None:
    names = [Contestant(n) for n in ("random", "catanatron:R", "catanatron:W", "catanatron:VP")]
    seats_by_name: dict[str, set[int]] = {}
    for seating in rotations(names):
        for seat, contestant in enumerate(seating):
            seats_by_name.setdefault(contestant.name, set()).add(seat)
    assert all(seats == {0, 1, 2, 3} for seats in seats_by_name.values())


def test_matches_are_deterministic() -> None:
    first, ref_a = play_match("m", 21, LINEUP)
    second, ref_b = play_match("m", 21, LINEUP)
    assert timeless(first) == timeless(second)
    assert ref_a.action_log == ref_b.action_log
    assert first.winner_seat is not None or first.truncated


def test_replay_reproduces_the_match_exactly() -> None:
    result, referee = play_match("m", 22, LINEUP)
    record = json.loads(json.dumps(replay_record(result, referee)))  # through real JSON
    rebuilt = replay(record)
    assert rebuilt.events == referee.events
    original_state = referee.adapter.referee_game.state
    rebuilt_state = rebuilt.adapter.referee_game.state
    assert rebuilt_state.player_state == original_state.player_state
    assert rebuilt.adapter.winner_seat == result.winner_seat


def test_privileged_bots_are_labeled() -> None:
    result, _ = play_match("m", 23, LINEUP)
    assert result.privileged_seats == (False, True, True, False)


def test_unknown_contestants_fail_fast() -> None:
    with pytest.raises(ValueError):
        Contestant("nonsense")
    with pytest.raises(ValueError):
        Contestant("catanatron:NOPE")


class IllegalAgent:
    def __init__(self, seed: str) -> None:
        pass

    def decide(self, observation: PlayerObservation) -> CanonicalAction:
        return CanonicalAction(ActionKind.BUILD_CITY, observation.seat, (999,))


def test_illegal_agent_actions_are_recorded_as_errors(monkeypatch: Any) -> None:
    monkeypatch.setitem(AGENTS, "illegal", IllegalAgent)
    lineup = [Contestant("illegal")] + [Contestant("random")] * 3
    result, _ = play_match("m", 24, lineup)
    assert result.error is not None and "illegal" in result.error
    assert result.winner_seat is None


def test_tournament_writes_results_replays_and_summary(tmp_path: Path) -> None:
    results = run_tournament(LINEUP, boards=1, base_seed=30, out_dir=tmp_path)
    assert len(results) == 4
    lines = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert {json.loads(line)["seed"] for line in lines} == {30}
    assert len(list((tmp_path / "replays").glob("*.json"))) == 4
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["matches"] == 4
    assert summary["contestants"]["random"]["seats_played"] == 8
    assert summary["contestants"]["catanatron:F"]["privileged"] is True
    assert summary["config"]["engine_commit"] != "unknown"


def test_cli_runs(tmp_path: Path, capsys: Any) -> None:
    code = arena_main(
        ["--lineup", "random,random,random,random", "--boards", "1", "--out", str(tmp_path)]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["matches"] == 4


def test_random_trader_completes_lawful_trades() -> None:
    lineup = [Contestant("random-trader")] + [Contestant("catanatron:R")] * 3
    result, referee = play_match("m", 25, lineup)
    assert result.error is None
    kinds = [e.event_type for e in referee.events]
    assert kinds.count(ActionKind.OFFER_TRADE) > 10
    assert ActionKind.CONFIRM_TRADE in kinds


def test_trade_probe_cli(capsys: Any) -> None:
    from catan_ai.cli.trade_probe import main as probe_main

    assert probe_main(["--bots", "catanatron:R", "--boards", "1"]) == 0
    report = json.loads(capsys.readouterr().out)["catanatron:R"]
    assert report["matches"] == 4 and report["errors"] == 0 and report["bot_offers"] == 0


def test_parallel_tournament_matches_serial() -> None:
    lineup = [Contestant("random"), Contestant("catanatron:W")] * 2
    serial = run_tournament(lineup, boards=2, base_seed=40)
    parallel = run_tournament(lineup, boards=2, base_seed=40, workers=3)
    assert [timeless(r) for r in serial] == [timeless(r) for r in parallel]
