"""Tournament bracket structure: seeds, slots, and field validation.

This module turns Kaggle's ``MNCAATourneySeeds`` / ``MNCAATourneySlots`` pair
into an explicit bracket graph.  Nothing here simulates anything - it answers
"who plays whom", so the simulator can be written once and tested against real
tournaments.

Why this is not derivable from the game log
-------------------------------------------
Tournament results say which teams *met*, not which slot they occupied.  To
replay a bracket you need the pairing logic, because a 1-seed losing in R1
changes who appears in the Elite Eight.  ``MNCAATourneySlots`` supplies that:
each ``Slot`` names the two slots or seeds feeding it.

Slot naming
-----------
``R<round><region><index>`` - e.g. ``R1W1``, ``R5W1``.  Round 1 slots reference
team seeds (``W01`` vs ``W16``); every later round references *slots*
(``R2W1`` consumes winners of ``R1W1`` and ``R1W8``).  Play-in seeds carry a
fourth character (``W16a``, ``W16b``) and feed a round-0 slot.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data_loader import DataSchemaError, _resolve

__all__ = [
    "SEEDS_FILENAME",
    "SLOTS_FILENAME",
    "TOURNEY_FILENAME",
    "REGIONS",
    "TournamentBracket",
    "load_seeds",
    "load_slots",
    "load_bracket",
    "load_tourney_games",
    "seed_region",
    "seed_number",
    "round_of_slot",
]

SEEDS_FILENAME = "MNCAATourneySeeds.csv"
SLOTS_FILENAME = "MNCAATourneySlots.csv"
TOURNEY_FILENAME = "MNCAATourneyDetailedResults.csv"

#: Regions are identified by letter rather than by name, so a bracket from any
#: season can be compared with any other.  Names change year to year; letters
#: are the stable key.
REGIONS: tuple[str, ...] = ("W", "X", "Y", "Z")

_SEED_RE = re.compile(r"^(?P<region>[WXYZ])(?P<num>\d{2})(?P<playin>[abAB])?$")
# Slot tags are not uniform: early rounds are R<r><region><n> (R1W1), the two
# semifinals are R5WX / R5YZ, and the final is R6CH.  So the tag is opaque
# rather than region-anchored.
_SLOT_RE = re.compile(r"^R(?P<round>[1-6])(?P<tag>[0-9A-Z]+)$")
_PLAYIN_SLOT_RE = re.compile(r"^(?P<region>[WXYZ])(?P<num>\d{2})$")


def seed_region(seed: str) -> str:
    """Region letter for a seed such as ``W01`` or ``Z16a``."""
    m = _SEED_RE.match(seed)
    if not m:
        raise DataSchemaError(f"malformed seed {seed!r}; expected e.g. 'W01' or 'Z16a'")
    return str(m["region"])


def seed_number(seed: str) -> int:
    """Numeric seed within its region (1-16), ignoring any play-in suffix."""
    m = _SEED_RE.match(seed)
    if not m:
        raise DataSchemaError(f"malformed seed {seed!r}; expected e.g. 'W01' or 'Z16a'")
    return int(m["num"])


def round_of_slot(slot: str) -> int:
    """Round number for a slot name.

    Returns:
        0 for the play-in pseudo-round (slots named like ``X16``), 1-6
        otherwise.  Round 5 is the Final Four and round 6 the championship.
    """
    m = _SLOT_RE.match(slot)
    if m:
        return int(m["round"])
    if _PLAYIN_SLOT_RE.match(slot):
        return 0
    raise DataSchemaError(
        f"malformed slot {slot!r}; expected e.g. 'R1W1', 'R5WX', 'R6CH' or 'X16'"
    )


def _read(path: str | Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(_resolve(path, filename))


def load_seeds(path: str | Path, season: int) -> dict[str, int]:
    """Seed string -> TeamID for one season.

    Args:
        path: Directory containing ``MNCAATourneySeeds.csv``.
        season: Championship year, e.g. 2025.

    Returns:
        Mapping of seed to team id.  Includes play-in seeds when present, so
        68 entries in recent seasons and 64 before the play-in era.
    """
    frame = _read(path, SEEDS_FILENAME)
    required = {"Season", "Seed", "TeamID"}
    missing = required - set(frame.columns)
    if missing:
        raise DataSchemaError(f"{SEEDS_FILENAME} missing column(s): {', '.join(sorted(missing))}")
    rows = frame.loc[frame["Season"] == season]
    if rows.empty:
        raise DataSchemaError(f"no seeds for season {season}")
    seeds: dict[str, int] = {}
    for seed, team in zip(rows["Seed"], rows["TeamID"], strict=True):
        text = str(seed).strip().upper()
        tid = int(team)
        if text in seeds:
            raise DataSchemaError(f"season {season} has duplicate seed {text}")
        seeds[text] = tid
    return seeds


def load_slots(path: str | Path, season: int) -> dict[str, tuple[str, str]]:
    """Bracket pairing graph for one season.

    Returns:
        ``{slot: (strong_side, weak_side)}`` where each side is either a team
        seed (``W01``) or another slot (``R1W1``).
    """
    frame = _read(path, SLOTS_FILENAME)
    required = {"Season", "Slot", "StrongSeed", "WeakSeed"}
    missing = required - set(frame.columns)
    if missing:
        raise DataSchemaError(f"{SLOTS_FILENAME} missing column(s): {', '.join(sorted(missing))}")
    rows = frame.loc[frame["Season"] == season]
    if rows.empty:
        raise DataSchemaError(f"no bracket slots for season {season}")
    slots: dict[str, tuple[str, str]] = {}
    for slot, strong, weak in zip(
        rows["Slot"], rows["StrongSeed"], rows["WeakSeed"], strict=True
    ):
        key = str(slot).strip().upper()
        if key in slots:
            raise DataSchemaError(f"season {season} has duplicate slot {key}")
        slots[key] = (str(strong).strip().upper(), str(weak).strip().upper())
    return slots


@dataclass(frozen=True)
class TournamentBracket:
    """A validated bracket for one season."""

    season: int
    seeds: Mapping[str, int]
    slots: Mapping[str, tuple[str, str]]

    @property
    def field_size(self) -> int:
        """Number of teams, counting both sides of any play-in game."""
        return len(self.seeds)

    @property
    def has_play_in(self) -> bool:
        return any(seed[-1] in ("a", "b", "A", "B") for seed in self.seeds)

    def rounds(self) -> dict[int, list[str]]:
        """Slots grouped by round, ascending.  Round 0 is the play-in."""
        grouped: dict[int, list[str]] = {}
        for slot in self.slots:
            grouped.setdefault(round_of_slot(slot), []).append(slot)
        for value in grouped.values():
            value.sort()
        return dict(sorted(grouped.items()))

    def validate(self) -> None:
        """Check the pairing graph is internally consistent.

        Raises:
            DataSchemaError: If a slot references an unknown seed or slot, if a
                non-champion slot is referenced by nothing, or if the field
                size is not one of the historically valid counts.
        """
        if self.field_size not in (64, 65, 66, 68):
            # 64 = no play-in; 65 = one play-in game (2003-2010 had exactly one,
            # in region X); 68 = four play-in games (2011 onward).  66 is
            # included defensively but has no known instance.
            raise DataSchemaError(
                f"season {self.season} has {self.field_size} seeds; "
                "expected 64, 65, 66 or 68"
            )
        for slot, (strong, weak) in self.slots.items():
            for side in (strong, weak):
                if side not in self.slots and side not in self.seeds:
                    raise DataSchemaError(
                        f"season {self.season} slot {slot} references "
                        f"{side!r}, which is neither a seed nor a slot"
                    )
        # Every slot except the championship must feed a later round, otherwise
        # a game would be simulated with nowhere for its winner to go.
        referenced = {s for pair in self.slots.values() for s in pair if s in self.slots}
        dangling = sorted(set(self.slots) - referenced)
        if len(dangling) > 1:
            raise DataSchemaError(
                f"season {self.season} has {len(dangling)} unconnected slot(s): "
                f"{', '.join(dangling[:5])}"
            )
        regions = {seed_region(s) for s in self.seeds}
        if not regions <= set(REGIONS):
            raise DataSchemaError(
                f"season {self.season} has unexpected region letter(s): "
                f"{', '.join(sorted(regions - set(REGIONS)))}"
            )


def load_bracket(path: str | Path, season: int) -> TournamentBracket:
    """Load and validate the bracket for one season."""
    bracket = TournamentBracket(
        season=season,
        seeds=load_seeds(path, season),
        slots=load_slots(path, season),
    )
    bracket.validate()
    return bracket


def load_tourney_games(
    path: str | Path, *, seasons: tuple[int, ...] | None = None
) -> pd.DataFrame:
    """Load ``MNCAATourneyDetailedResults.csv``.

    Args:
        path: Directory containing the file, or the file itself.
        seasons: Optional season filter.

    Returns:
        A frame with the pipeline's canonical column names, sorted by
        ``(Season, DayNum)``.  Normalisation is shared with
        :func:`src.data_loader.load_games`, so Kaggle's ``WScore``/``WOR``
        spellings are accepted.
    """
    from src.data_loader import load_games  # local: avoids an import cycle

    return load_games(_resolve(path, TOURNEY_FILENAME), seasons=seasons)
