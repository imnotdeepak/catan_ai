"""Exact VP, city, road, army and dev timing (STR-019, plan §5.5).

Two kinds of test: random games through observations, where every accounting
fact is checked against the engine's private state at every decision, and
small fixtures that pin each threshold and tie rule.
"""

from typing import Any

import pytest
from catanatron import Action, ActionType
from catanatron.apply_action import apply_action
from catanatron.models.enums import ActionRecord
from catanatron.state_functions import build_city, buy_dev_card

from catan_ai.engine import ActionKind
from catan_ai.observations import DEV_CARDS, Referee
from catan_ai.opening.board import analyze
from catan_ai.victory.score import (
    VICTORY_POINT,
    bought_this_turn,
    own_points,
    pieces_left,
    playable_dev_cards,
    points_range,
    seat_scores,
    takes_award,
    takes_largest_army,
    takes_longest_road,
    wins,
)
from tests.observations.test_firewall import play
from tests.rules.helpers import force_turn, past_setup, set_hand

SEEDS = (3, 11, 19, 27)
PLAY_KINDS = {
    ActionKind.PLAY_KNIGHT: "KNIGHT",
    ActionKind.PLAY_YEAR_OF_PLENTY: "YEAR_OF_PLENTY",
    ActionKind.PLAY_MONOPOLY: "MONOPOLY",
    ActionKind.PLAY_ROAD_BUILDING: "ROAD_BUILDING",
}


def true_points(referee: Referee, seat: int) -> int:
    return int(referee.adapter.referee_game.state.player_state[f"P{seat}_ACTUAL_VICTORY_POINTS"])


@pytest.fixture(scope="module")
def sweep() -> dict[str, int]:
    """Play random games, checking accounting at every decision; return coverage counts."""
    seen = {"army_changes": 0, "road_changes": 0, "dev_plays": 0, "vp_cards": 0, "games": 0}
    for seed in SEEDS:
        previous: Any = None
        for step, referee in enumerate(play(seed)):
            adapter = referee.adapter
            if adapter.is_over:
                winner = adapter.winner_seat
                if winner is not None:
                    assert wins(true_points(referee, winner), winner, adapter.turn_seat)
                seen["games"] += 1
                break
            viewers = range(4) if step % 10 == 0 else (adapter.current_seat,)
            for viewer in viewers:
                obs = referee.observe(viewer)
                check_observation(referee, obs)
                seen["vp_cards"] += obs.own_dev_cards[VICTORY_POINT] > 0
            obs = referee.observe(adapter.current_seat)
            if obs.prompt == "PLAY_TURN":
                check_dev_timing(obs)
            if previous is not None:
                count_award_changes(previous, obs, seen)
            seen["dev_plays"] += any(a.kind in PLAY_KINDS for a in obs.legal_actions)
            previous = obs
    return seen


def check_observation(referee: Referee, obs: Any) -> None:
    scores = seat_scores(obs)
    assert own_points(obs) == true_points(referee, obs.seat)
    for player, score in zip(obs.players, scores, strict=True):
        assert score.public_points == player.visible_victory_points
        low, high = points_range(obs, player.seat)
        assert low <= true_points(referee, player.seat) <= high
        left = pieces_left(obs, player.seat)
        assert left == {
            "roads": player.roads_available,
            "settlements": player.settlements_available,
            "cities": player.cities_available,
        }
        # No one is ever on 10 at a decision of their own turn: that would have ended the game.
        assert not wins(true_points(referee, player.seat), player.seat, obs.turn_seat)
    assert sum(s.largest_army for s in scores) <= 1
    assert sum(s.longest_road for s in scores) <= 1
    for s in scores:
        if s.largest_army:
            assert s.knights_played >= 3
            assert all(o.knights_played <= s.knights_played for o in scores)


def check_dev_timing(obs: Any) -> None:
    """Our playable cards agree with the engine's legal actions on our turn."""
    playable = playable_dev_cards(obs)
    legal = {a.kind for a in obs.legal_actions}
    for kind, card in PLAY_KINDS.items():
        can = playable[DEV_CARDS.index(card)] > 0
        if kind in legal:
            assert can, f"engine allows {kind} but accounting says no"
        elif card in ("KNIGHT", "MONOPOLY"):  # no further card-specific conditions
            assert not can, f"accounting allows {kind} but engine does not"


def count_award_changes(before: Any, after: Any, seen: dict[str, int]) -> None:
    old, new = seat_scores(before), seat_scores(after)
    for flag, key in (("largest_army", "army_changes"), ("longest_road", "road_changes")):
        was = next((s.seat for s in old if getattr(s, flag)), None)
        now = next((s.seat for s in new if getattr(s, flag)), None)
        if was != now:
            seen[key] += 1
            if flag == "largest_army" and now is not None:
                gained = new[now].knights_played
                assert gained >= 3 and (was is None or gained > old[was].knights_played)


def test_random_games_reconcile_with_the_engine(sweep: dict[str, int]) -> None:
    assert sweep["games"] > 0  # at least one game reached a winner
    assert sweep["road_changes"] > 0 and sweep["dev_plays"] > 0  # the checks were exercised


# ----- Fixtures


def test_award_threshold_and_tie_rules() -> None:
    assert not takes_award(2, 3, None)
    assert takes_award(3, 3, None)
    assert not takes_award(3, 3, 3)  # a tie leaves the award with the holder
    assert takes_award(4, 3, 3)
    assert not takes_award(4, 5, None)
    assert takes_award(5, 5, None)
    assert not takes_award(6, 5, 6)


def test_largest_army_engine_threshold_and_transfer() -> None:
    game = past_setup(4)
    state = game.state

    def knight(seat: int) -> None:
        key = f"P{seat}"
        state.player_state[f"{key}_KNIGHT_IN_HAND"] = 1
        state.player_state[f"{key}_KNIGHT_OWNED_AT_START"] = True
        state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = False
        force_turn(game, seat)
        action = Action(state.colors[seat], ActionType.PLAY_KNIGHT_CARD, None)
        apply_action(state, action, ActionRecord(action, None))

    points = [state.player_state[f"P{s}_ACTUAL_VICTORY_POINTS"] for s in range(4)]
    knight(0)
    knight(0)
    assert not state.player_state["P0_HAS_ARMY"]
    knight(0)  # third knight: Largest Army, 2 points
    assert state.player_state["P0_HAS_ARMY"]
    assert state.player_state["P0_ACTUAL_VICTORY_POINTS"] == points[0] + 2
    for _ in range(3):
        knight(1)  # 3 against 3: a tie, the holder keeps it
    assert state.player_state["P0_HAS_ARMY"] and not state.player_state["P1_HAS_ARMY"]
    knight(1)  # 4 against 3: transfers
    assert state.player_state["P1_HAS_ARMY"] and not state.player_state["P0_HAS_ARMY"]
    assert state.player_state["P0_ACTUAL_VICTORY_POINTS"] == points[0]
    assert state.player_state["P1_ACTUAL_VICTORY_POINTS"] == points[1] + 2


def test_vp_card_counts_at_once_and_wins_on_own_turn() -> None:
    game = past_setup(9)
    state = game.state
    force_turn(game, 0)
    visible = state.player_state["P0_VICTORY_POINTS"]
    state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 9
    set_hand(game, 0, (0, 0, 1, 1, 1))
    buy_dev_card(state, state.colors[0], "VICTORY_POINT")
    assert state.player_state["P0_ACTUAL_VICTORY_POINTS"] == 10
    assert state.player_state["P0_VICTORY_POINTS"] == visible  # still hidden
    assert game.winning_color() == state.colors[0]


def first_turn_referee(seed: int) -> Referee:
    """A referee at the first decision after setup."""
    for referee in play(seed):
        if not referee.observe(referee.adapter.current_seat).is_initial_build_phase:
            return referee
    raise AssertionError("game ended during setup")


def test_city_upgrade_returns_the_settlement_piece() -> None:
    referee = first_turn_referee(5)
    seat = referee.adapter.current_seat
    state = referee.adapter.referee_game.state
    node = next(n for n, s, kind in referee.observe(seat).board.buildings if s == seat)
    before = referee.observe(seat)
    set_hand(referee.adapter.referee_game, seat, (0, 0, 0, 2, 3))
    build_city(state, state.colors[seat], node)
    state.board.build_city(state.colors[seat], node)
    after = referee.observe(seat)
    assert pieces_left(after, seat)["settlements"] == pieces_left(before, seat)["settlements"] + 1
    assert pieces_left(after, seat)["cities"] == pieces_left(before, seat)["cities"] - 1
    assert own_points(after) == own_points(before) + 1
    assert seat_scores(after)[seat].cities == 1


def test_bought_cards_wait_a_turn_and_one_card_per_turn() -> None:
    for referee in play(11):
        obs = referee.observe(referee.adapter.current_seat)
        buys = [a for a in obs.legal_actions if a.kind == ActionKind.BUY_DEV_CARD]
        if obs.prompt != "PLAY_TURN" or not buys:
            continue
        held = obs.own_dev_cards
        referee.execute(buys[0])
        after = referee.observe(obs.seat)
        new = bought_this_turn(after)
        assert sum(new) == 1
        assert playable_dev_cards(after) == tuple(
            0 if i == VICTORY_POINT else h for i, h in enumerate(held)
        )
        return
    pytest.fail("no development card was bought")


def test_road_award_prediction_matches_the_engine_on_new_roads() -> None:
    checked = awarded = 0
    for seed in SEEDS:
        for referee in play(seed):
            obs = referee.observe(referee.adapter.current_seat)
            if obs.prompt != "PLAY_TURN":
                continue
            roads = [a for a in obs.legal_actions if a.kind == ActionKind.BUILD_ROAD]
            if not roads:
                continue
            edge = roads[0].params[0]
            analysis = analyze(obs)
            if _touches_opponent_building(obs, analysis.road_owner, edge):
                continue  # engine convention differs there (rules matrix Q-05)
            length = analysis.longest_road_after(obs.seat, edge)
            predicted = takes_longest_road(obs, obs.seat, length)
            referee.execute(roads[0])
            assert referee.observe(obs.seat).players[obs.seat].has_longest_road == predicted
            checked += 1
            awarded += predicted
            if referee.adapter.is_over:  # the road may have won the game
                break
    assert checked >= 40 and awarded > 0


def _touches_opponent_building(obs: Any, roads: dict[Any, int], edge: Any) -> bool:
    seat = obs.seat
    own_nodes = {n for e, s in {**roads, tuple(edge): seat}.items() if s == seat for n in e}
    return any(n in own_nodes and s != seat for n, s, _ in obs.board.buildings)


def test_army_prediction_matches_the_engine_on_played_knights() -> None:
    checked = 0
    for seed in SEEDS:
        for referee in play(seed):
            obs = referee.observe(referee.adapter.current_seat)
            knights = [a for a in obs.legal_actions if a.kind == ActionKind.PLAY_KNIGHT]
            if not knights:
                continue
            count = obs.players[obs.seat].played_dev_cards[DEV_CARDS.index("KNIGHT")] + 1
            predicted = takes_largest_army(obs, obs.seat, count)
            referee.execute(knights[0])
            assert referee.observe(obs.seat).players[obs.seat].has_largest_army == predicted
            checked += 1
            if referee.adapter.is_over:  # Largest Army may have won the game
                break
    assert checked > 0
