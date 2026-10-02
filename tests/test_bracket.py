"""Tests for Kaggle schema normalisation and tournament bracket loading."""

from __future__ import annotations

import pandas as pd
import pytest

from src.bracket import (
    REGIONS,
    TournamentBracket,
    load_bracket,
    load_seeds,
    load_slots,
    round_of_slot,
    seed_number,
    seed_region,
)
from src.data_loader import DataSchemaError, load_games

# A miniature file in Kaggle's spelling: WScore/LScore/WOR/LOR rather than the
# canonical WTeamScore/LTeamScore/WOREB/LOREB.
KAGGLE_ROWS = pd.DataFrame(
    {
        "Season": [2025, 2025],
        "DayNum": [1, 2],
        "WTeamID": [1001, 1002],
        "WScore": [80, 71],
        "LTeamID": [1003, 1004],
        "LScore": [72, 66],
        "WLoc": ["H", "N"],
        "NumOT": [0, 0],
        "WFGA": [60, 55],
        "WFGA3": [24, 20],
        "WFTA": [20, 22],
        "WOR": [11, 9],
        "WTO": [12, 14],
        "LFGA": [58, 60],
        "LFGA3": [18, 15],
        "LFTA": [14, 18],
        "LOR": [9, 13],
        "LTO": [15, 11],
    }
)

# Same games in the NCAA API spelling.
NCAA_ROWS = KAGGLE_ROWS.rename(
    columns={"WScore": "WTeamScore", "LScore": "LTeamScore",
             "WOR": "WOREB", "LOR": "LOREB"}
)


def _write(tmp_path, frame, name="MRegularSeasonDetailedResults.csv"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tmp_path / name, index=False)
    return tmp_path


class TestColumnNormalisation:
    def test_kaggle_spelling_is_accepted(self, tmp_path) -> None:
        games = load_games(_write(tmp_path, KAGGLE_ROWS))
        assert len(games) == 2
        assert games.loc[0, "WTeamScore"] == 80
        assert games.loc[0, "WOREB"] == 11

    def test_ncaa_spelling_is_accepted(self, tmp_path) -> None:
        games = load_games(_write(tmp_path, NCAA_ROWS))
        assert games.loc[0, "WTeamScore"] == 80
        assert games.loc[0, "WOREB"] == 11

    def test_both_spellings_yield_identical_frames(self, tmp_path) -> None:
        a = load_games(_write(tmp_path / "a", KAGGLE_ROWS))
        b = load_games(_write(tmp_path / "b", NCAA_ROWS))
        pd.testing.assert_frame_equal(a, b)

    def test_missing_column_still_raises(self, tmp_path) -> None:
        broken = KAGGLE_ROWS.drop(columns=["WFGA3"])
        with pytest.raises(DataSchemaError, match="WFGA3"):
            load_games(_write(tmp_path, broken))


class TestSeedHelpers:
    @pytest.mark.parametrize(
        "seed,region,number",
        [
            ("W01", "W", 1),
            ("X16", "X", 16),
            ("Z16a", "Z", 16),
            ("W12B", "W", 12),
        ],
    )
    def test_parses(self, seed: str, region: str, number: int) -> None:
        assert seed_region(seed) == region
        assert seed_number(seed) == number

    def test_rejects_malformed(self) -> None:
        with pytest.raises(DataSchemaError):
            seed_region("A01")
        with pytest.raises(DataSchemaError):
            seed_number("W1")

    @pytest.mark.parametrize(
        "slot,round_no",
        [("R1W1", 1), ("R2W8", 2), ("R5WX", 5), ("R5YZ", 5), ("R6CH", 6), ("X16", 0)],
    )
    def test_round_of_slot(self, slot: str, round_no: int) -> None:
        assert round_of_slot(slot) == round_no

    def test_round_of_slot_rejects_junk(self) -> None:
        with pytest.raises(DataSchemaError):
            round_of_slot("R7W1")


def _seed_frame(season: int = 2025) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        for n in range(1, 17):
            rows.append({"Season": season, "Seed": f"{region}{n:02d}",
                         "TeamID": 1000 + len(rows)})
    return pd.DataFrame(rows)


def _slot_frame(season: int = 2025) -> pd.DataFrame:
    """A minimal but structurally valid 64-team bracket."""
    rows = []
    for region in REGIONS:
        for i in range(1, 9):
            rows.append({"Season": season, "Slot": f"R1{region}{i}",
                         "StrongSeed": f"{region}{(i - 1) * 2 + 1:02d}",
                         "WeakSeed": f"{region}{(i - 1) * 2 + 2:02d}"})
    # R2: pairs of R1 within a region.  R4 pairs R3.  R5 = Final Four, R6 = final.
    for region in REGIONS:
        for i in range(1, 5):
            rows.append({"Season": season, "Slot": f"R2{region}{i}",
                         "StrongSeed": f"R1{region}{(i - 1) * 2 + 1}",
                         "WeakSeed": f"R1{region}{(i - 1) * 2 + 2}"})
        rows.append({"Season": season, "Slot": f"R3{region}1",
                     "StrongSeed": f"R2{region}1", "WeakSeed": f"R2{region}2"})
        rows.append({"Season": season, "Slot": f"R3{region}2",
                     "StrongSeed": f"R2{region}3", "WeakSeed": f"R2{region}4"})
        rows.append({"Season": season, "Slot": f"R4{region}1",
                     "StrongSeed": f"R3{region}1", "WeakSeed": f"R3{region}2"})
    rows.append({"Season": season, "Slot": "R5WX", "StrongSeed": "R4W1", "WeakSeed": "R4X1"})
    rows.append({"Season": season, "Slot": "R5YZ", "StrongSeed": "R4Y1", "WeakSeed": "R4Z1"})
    rows.append({"Season": season, "Slot": "R6CH", "StrongSeed": "R5WX", "WeakSeed": "R5YZ"})
    return pd.DataFrame(rows)


@pytest.fixture
def bracket_dir(tmp_path):
    _seed_frame().to_csv(tmp_path / "MNCAATourneySeeds.csv", index=False)
    _slot_frame().to_csv(tmp_path / "MNCAATourneySlots.csv", index=False)
    return tmp_path


class TestBracketLoading:
    def test_loads_and_validates(self, bracket_dir) -> None:
        b = load_bracket(bracket_dir, 2025)
        assert b.field_size == 64
        assert b.has_play_in is False
        assert len(b.slots) == 32 + 16 + 8 + 4 + 2 + 1

    def test_round_counts(self, bracket_dir) -> None:
        rounds = load_bracket(bracket_dir, 2025).rounds()
        assert [len(rounds[r]) for r in sorted(rounds)] == [32, 16, 8, 4, 2, 1]

    def test_seeds_and_slots_are_season_scoped(self, bracket_dir) -> None:
        seeds = load_seeds(bracket_dir, 2025)
        assert len(seeds) == 64
        assert seeds["W01"] == 1000
        assert len(load_slots(bracket_dir, 2025)) == 63
        with pytest.raises(DataSchemaError, match="no seeds"):
            load_seeds(bracket_dir, 1999)

    def test_rejects_unreachable_reference(self, bracket_dir) -> None:
        slots = _slot_frame()
        slots.loc[slots.Slot == "R2W1", "StrongSeed"] = "R1ZZ9"
        slots.to_csv(bracket_dir / "MNCAATourneySlots.csv", index=False)
        with pytest.raises(DataSchemaError, match="neither a seed nor a slot"):
            load_bracket(bracket_dir, 2025)

    def test_rejects_impossible_field_size(self, bracket_dir) -> None:
        seeds = _seed_frame().head(40)
        seeds.to_csv(bracket_dir / "MNCAATourneySeeds.csv", index=False)
        with pytest.raises(DataSchemaError, match="expected 64, 65, 66 or 68"):
            load_bracket(bracket_dir, 2025)

    def test_rejects_duplicate_seed(self, bracket_dir) -> None:
        seeds = _seed_frame()
        seeds.loc[len(seeds)] = {"Season": 2025, "Seed": "W01", "TeamID": 9999}
        seeds.to_csv(bracket_dir / "MNCAATourneySeeds.csv", index=False)
        with pytest.raises(DataSchemaError, match="duplicate seed"):
            load_bracket(bracket_dir, 2025)


def _play_in_seeds() -> pd.DataFrame:
    """The X16 slot replaced by two play-in sides, X16a/X16b."""
    return pd.DataFrame(
        {
            "Season": [2025, 2025],
            "Seed": ["X16a", "X16b"],
            "TeamID": [9001, 9002],
        }
    )


def _single_play_in_dir(bracket_dir) -> None:
    """Rewrite the fixture as a 2003-2010 style bracket: one play-in game.

    Mirrors the real 2003 file exactly: ``X16`` becomes a slot whose sides are
    the seeds ``X16a``/``X16b``, and ``R1X1`` then plays ``X01`` against that
    slot's winner.
    """
    base = _seed_frame()
    base = base[base.Seed != "X16"]
    pd.concat([base, _play_in_seeds()], ignore_index=True).to_csv(
        bracket_dir / "MNCAATourneySeeds.csv", index=False
    )
    slots = _slot_frame()
    slots.loc[slots.Slot == "R1X8", "WeakSeed"] = "X16"
    slots = pd.concat(
        [
            slots,
            pd.DataFrame(
                [{"Season": 2025, "Slot": "X16",
                  "StrongSeed": "X16a", "WeakSeed": "X16b"}]
            ),
        ],
        ignore_index=True,
    )
    slots.to_csv(bracket_dir / "MNCAATourneySlots.csv", index=False)


def test_play_in_field_is_65_when_single_game(bracket_dir) -> None:
    """2003-2010 had exactly one play-in game, so 65 entries is correct.

    X16 is replaced by the two play-in sides X16a/X16b, and their winner
    takes the X16 slot.  That is 64 - 1 + 2 = 65 teams.
    """
    _single_play_in_dir(bracket_dir)
    b = load_bracket(bracket_dir, 2025)
    assert b.field_size == 65
    assert b.has_play_in is True


def test_play_in_slot_feeds_round_one(bracket_dir) -> None:
    """The play-in winner must occupy a real slot, so R1X8 reads 'X16'."""
    _single_play_in_dir(bracket_dir)
    b = load_bracket(bracket_dir, 2025)
    assert b.slots["X16"] == ("X16A", "X16B")
    assert "X16" in b.slots["R1X8"]
    assert 0 in b.rounds()
    assert len(b.rounds()[0]) == 1


class TestBracketStructure:
    def test_champion_slot_is_the_only_terminal_one(self, bracket_dir) -> None:
        b = TournamentBracket(2025, load_seeds(bracket_dir, 2025),
                              load_slots(bracket_dir, 2025))
        referenced = {s for pair in b.slots.values() for s in pair if s in b.slots}
        assert set(b.slots) - referenced == {"R6CH"}

    def test_every_team_reaches_the_championship_slot(self, bracket_dir) -> None:
        """Each seed must resolve forward to a single terminal slot.

        This is the property the simulator relies on: walking the graph from
        any team must terminate at the championship, with no orphan branch.
        """
        b = TournamentBracket(2025, load_seeds(bracket_dir, 2025),
                              load_slots(bracket_dir, 2025))
        consumers: dict[str, list[str]] = {}
        for slot, pair in b.slots.items():
            for side in pair:
                consumers.setdefault(side, []).append(slot)

        for seed in b.seeds:
            # A seed is a team, not a game.  Its first game is the slot that
            # lists it, and it is eliminated in round 1, so the surviving
            # path is the *other* branch of that game.
            assert seed in consumers, f"seed {seed} never appears in any slot"
            entry = consumers[seed][0]
            strong, weak = b.slots[entry]
            path = weak if strong == seed else strong
            node, hops, seen = entry, 1, {entry, path}
            while node != "R6CH":
                assert node in consumers, f"{node} is a dead end feeding nothing"
                assert len(consumers[node]) == 1, (
                    f"{node} feeds {len(consumers[node])} slots, expected 1"
                )
                nxt = consumers[node][0]
                assert nxt not in seen, f"cycle reached from {seed}"
                seen.add(nxt)
                node = nxt
                hops += 1
            assert hops == 6, f"{seed} took {hops} hops to the championship"
