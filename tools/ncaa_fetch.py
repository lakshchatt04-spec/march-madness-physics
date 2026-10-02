"""Build Kaggle-shaped MBB CSV files from the public NCAA.com JSON endpoints.

Sources (no API key, no auth):
  * schedule / results : https://data.ncaa.com/casablanca/scoreboard/basketball-men/d1/<YYYY>/<MM>/<DD>/scoreboard.json
  * team box scores    : https://ncaa-api.henrygd.me/game/<contestId>/boxscore

The second host is a free volunteer-run mirror of ncaa.com (MIT licensed,
https://github.com/henrygd/ncaa-api).  It documents a 5 req/s per-IP limit, so
this script stays under that and caches every response on disk to make re-runs
free.

Outputs, in the exact column layout the project's ``data_loader`` validates:
  MRegularSeasonDetailedResults.csv
  MTeams.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

SCOREBOARD = (
    "https://data.ncaa.com/casablanca/scoreboard/basketball-men/d1/{y}/{m:02d}/{d:02d}/scoreboard.json"
)
BOXSCORE = "https://ncaa-api.henrygd.me/game/{cid}/boxscore"

# Venues that are not a team's home court.  Games here are treated as neutral
# so the dyadic fit does not mistake a conference tournament for home court.
NEUTRAL_VENUES = {
    "levi's stadium", "at&t stadium", "state farm stadium", "lucas oil stadium",
    "rocket mortgage fieldhouse", "rocket mortgage arena", "gainbridge fieldhouse",
    "t-mobile arena", "t-mobile center", "ppg paints arena", "ppg paints fieldhouse",
    "spectrum center", "spectrum arena", "mortgage matchup center", "chase center",
    "barclays center", "moda center", "climate pledge arena", "climate pledge center",
    "bon secours arena", "bon secours wellness arena", "american airlines center",
    "united center", "capital one arena", "capital one center", "fiserv forum",
    "msg arena", "msg sphere", "little caesars arena", "little caesars center",
    "miller park", "american family field", "target center", "enterprise center",
    "pnc arena", "pnc center", "smoothie king center", "charlotte center",
    "fedexforum", "fedex forum", "kia center", "kia forum", " Kia arena",
    "ball arena", "ball arena parking", "paycom center", "paycom center parking",
    "crypto.com arena", "crypto.com center", "footprint center", "delta center",
    "crown center", "tucson arena", "sfo fighters stadium", "citi field",
}

GAME_COLUMNS = [
    "Season", "DayNum", "WTeamID", "WTeamScore", "LTeamID", "LTeamScore", "WLoc",
    "WFGA", "WFGA3", "WFTA", "WOREB", "WTO",
    "LFGA", "LFGA3", "LFTA", "LOREB", "LTO",
]

_UA = {"User-Agent": "cbb-dataset-builder/1.0 (educational; contact: local)"}
_lock = threading.Lock()
_last = [0.0]
MIN_INTERVAL = 0.21  # ~4.8 req/s, just under the documented 5 req/s per-IP limit

#: Why games were dropped.  Silent loss is the failure mode this whole scraper
#: was vulnerable to: a run can "succeed", report every game as processed, and
#: still emit an almost-empty dataset.  Every rejection path must land here.
_STATS: Counter[str] = Counter()


def _throttle() -> None:
    with _lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()


def _get(url: str, cache_dir: Path | None, attempts: int = 4) -> object | None:
    key = re.sub(r"[^A-Za-z0-9]+", "_", url) + ".json"
    path = cache_dir / key if cache_dir else None
    if path and path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
    for attempt in range(attempts):
        _throttle()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=45) as r:
                raw = r.read().decode("utf-8")
            if path:
                path.write_text(raw, encoding="utf-8")
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            # A 404 is a *permanent* statement: this game has no box score.
            # A 502/503 is the upstream backend struggling, which is transient
            # and routinely succeeds on a later attempt.  Collapsing the two
            # into "return None" is what silently produced an empty 2021
            # season while the progress log cheerfully reported success.
            if exc.code == 404:
                _STATS["http_404_no_data"] += 1
                return None
            if exc.code in (502, 503, 504):
                _STATS[f"http_{exc.code}_upstream"] += 1
                time.sleep(1.5 * (attempt + 1))
                continue
            _STATS[f"http_{exc.code}"] += 1
            time.sleep(1.5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            _STATS["transport_error"] += 1
            time.sleep(1.5 * (attempt + 1))
    _STATS["gave_up"] += 1
    return None


def season_dates(year: int) -> list[date]:
    """Nov 1 (year-1) through Apr 10 (year) inclusive."""
    start, end = date(year - 1, 11, 1), date(year, 4, 10)
    out, cur = [], start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _parse_date(raw: str) -> date | None:
    """Scoreboard dates look like ``01-06-2024``; fall back to request day."""
    m = re.match(r"^(\d{2})-(\d{2})-(\d{4})$", (raw or "").strip())
    return date(int(m.group(3)), int(m.group(1)), int(m.group(2))) if m else None


def kaggle_season(day: date) -> int:
    """Kaggle labels a season by its ending calendar year.

    A November/December game belongs to the season that ends the following
    spring, so bump it; a January-April game already carries the end year.
    """
    return day.year + 1 if day.month >= 11 else day.year


def collect_games(year: int, cache_dir: Path, limit: int | None) -> list[dict]:
    """Enumerate every completed D-I men's game for the ``year`` season.

    The scoreboard payload carries scores and team *names* but no team IDs, so
    it is used only to enumerate contest IDs and dates; the box score is the
    single source of truth for IDs, venue flags and statistics.
    """
    games: list[dict] = []
    for day in season_dates(year):
        if limit is not None and len(games) >= limit:
            break
        payload = _get(SCOREBOARD.format(y=day.year, m=day.month, d=day.day), cache_dir)
        if not isinstance(payload, dict):
            continue
        for entry in payload.get("games") or []:
            g = entry.get("game") or {}
            if g.get("gameState") != "final" and g.get("finalMessage") != "FINAL":
                continue
            m = re.search(r"/game/(\d+)", g.get("url") or "")
            home, away = g.get("home") or {}, g.get("away") or {}
            if not m or not (home.get("score") and away.get("score")):
                continue
            played = _parse_date(g.get("startDate") or "") or day
            games.append({
                "contest_id": m.group(1),
                "date": played,
                "kaggle_season": kaggle_season(played),
                "scores": {int(home["score"]), int(away["score"])},
                "conference": ((home.get("conferences") or [{}])[0]).get("conferenceName", ""),
                # bracketId/bracketRound are populated only for NCAA tournament
                # games, which is exactly the split Kaggle draws between
                # MRegularSeasonDetailedResults.csv and MTourneyDetailedResults.csv.
                "is_tourney": bool(g.get("bracketId")),
            })
    return games


def _num(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def fetch_row(game: dict, cache_dir: Path) -> tuple[dict, dict[int, str], bool] | None:
    """Fetch one box score and emit a Kaggle-shaped row plus team-name records.

    Returns ``None`` - rather than raising - for any game that cannot be
    trusted: a missing box score, a stats block with a non-numeric field, a
    score that disagrees with the scoreboard, or a tie.  Those games are
    dropped so the dataset never contains a silently malformed row.
    """
    payload = _get(BOXSCORE.format(cid=game["contest_id"]), cache_dir)
    if not isinstance(payload, dict):
        _STATS["reject:no_payload"] += 1
        return None
    if str(payload.get("status", "")).upper() not in ("F", "FINAL"):
        _STATS[f"reject:status={payload.get('status')!r}"] += 1
        return None

    stats_by_id: dict[int, dict[str, int]] = {}
    for entry in payload.get("teamBoxscore") or []:
        stats = entry.get("teamStats") or {}
        tid = _num(entry.get("teamId"))
        if tid is None or not stats:
            continue
        parsed = {
            "FGA": _num(stats.get("fieldGoalsAttempted")),
            "FGA3": _num(stats.get("threePointsAttempted")),
            "FTA": _num(stats.get("freeThrowsAttempted")),
            "OREB": _num(stats.get("offensiveRebounds")),
            "TO": _num(stats.get("turnovers")),
            "PTS": _num(stats.get("points")),
        }
        if any(v is None for v in parsed.values()):
            _STATS["reject:non_numeric_stat"] += 1
            return None
        stats_by_id[tid] = {k: int(v) for k, v in parsed.items() if v is not None}

    meta = {
        int(t["teamId"]): t
        for t in (payload.get("teams") or [])
        if t.get("teamId")
    }
    usable = [tid for tid in meta if tid in stats_by_id]
    if len(usable) != 2:
        _STATS["reject:team_id_mismatch"] += 1
        return None
    home_ids = [tid for tid in usable if meta[tid].get("isHome") is True]
    if len(home_ids) != 1:
        _STATS[f"reject:isHome={sorted(repr(meta[t].get('isHome')) for t in usable)}"] += 1
        return None
    h = home_ids[0]
    a = next(tid for tid in usable if tid != h)

    if {stats_by_id[h]["PTS"], stats_by_id[a]["PTS"]} != game["scores"]:
        _STATS["reject:score_mismatch"] += 1
        return None  # box score disagrees with the scoreboard
    if stats_by_id[h]["PTS"] == stats_by_id[a]["PTS"]:
        _STATS["reject:tie"] += 1
        return None  # a tie has no winner

    win, lose = (h, a) if stats_by_id[h]["PTS"] > stats_by_id[a]["PTS"] else (a, h)
    # NCAA tournament sites are neutral, so isHome is not a venue signal there.
    wloc = "N" if game.get("is_tourney") else ("H" if win == h else "A")
    row = {
        "Season": game["kaggle_season"],
        "DayNum": game.get("daynum", 0),
        "WLoc": wloc,
        "WTeamID": win, "WTeamScore": stats_by_id[win]["PTS"],
        "LTeamID": lose, "LTeamScore": stats_by_id[lose]["PTS"],
    }
    for prefix, tid in (("W", win), ("L", lose)):
        row.update({
            f"{prefix}FGA": stats_by_id[tid]["FGA"],
            f"{prefix}FGA3": stats_by_id[tid]["FGA3"],
            f"{prefix}FTA": stats_by_id[tid]["FTA"],
            f"{prefix}OREB": stats_by_id[tid]["OREB"],
            f"{prefix}TO": stats_by_id[tid]["TO"],
        })
    names = {
        tid: (meta[tid].get("nameShort") or meta[tid].get("nameFull") or "")
        for tid in usable
    }
    return row, names, bool(game.get("is_tourney"))


def _report_stats() -> None:
    """Print every drop reason, so no game can vanish without explanation."""
    if not _STATS:
        return
    print("\n=== drop reasons ===")
    for reason, n in _STATS.most_common():
        print(f"  {n:7d}  {reason}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, action="append", default=None,
                    help="championship year, e.g. 2026. Repeatable.")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path(".ncaa_cache"))
    ap.add_argument("--limit", type=int, default=None, help="pilot: stop after N games")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-games", type=int, default=20,
                    help="drop teams below this many games, which filters out the "
                         "D-II/D-III programmes the d1 scoreboard endpoint leaks in")
    ap.add_argument("--min-acceptance", type=float, default=0.90,
                    help="abort if any season yields fewer than this fraction of its "
                         "games. Guards against upstream sources quietly dropping "
                         "whole seasons (the box-score API 502s on pre-2023 seasons).")
    args = ap.parse_args()

    seasons = args.season or [2025]
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    tourney_rows: list[dict] = []
    team_names: dict[int, str] = {}
    season_rates: list[tuple[int, int, int, float]] = []

    for season in seasons:
        print(f"[{season}] reading scoreboards...", flush=True)
        games = collect_games(season, args.cache, args.limit)
        if args.limit is not None:
            games = games[: args.limit]
        print(f"[{season}] {len(games)} games found; fetching box scores...", flush=True)
        dates = sorted({g["date"] for g in games})
        first = dates[0] if dates else None
        for g in games:
            # DayNum is days since the season's first game, per season.
            g["daynum"] = (g["date"] - first).days + 1 if first else 0

        done = 0
        accepted = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(lambda g: fetch_row(g, args.cache), games):
                done += 1
                if done % 250 == 0 or done == len(games):
                    print(f"[{season}] box scores {done}/{len(games)}", flush=True)
                if result is None:
                    continue
                row, names, is_tourney = result
                accepted += 1
                (tourney_rows if is_tourney else all_rows).append(row)
                team_names.update({k: v for k, v in names.items() if v})

        # The progress counter above counts ATTEMPTS.  Without this line a season
        # can report "6075/6075" and contribute almost nothing, which is exactly
        # how an empty 2021 slipped through unnoticed.
        rate = accepted / len(games) if games else 0.0
        print(f"[{season}] accepted {accepted}/{len(games)} ({rate:.1%})", flush=True)
        season_rates.append((season, len(games), accepted, rate))

    if not all_rows:
        print("no rows collected", file=sys.stderr)
        _report_stats()
        return 1

    # Data-integrity gate.  Write the dataset only if every season came through
    # intact; otherwise a half-empty file replaces a good one on disk and the
    # loss is invisible until someone notices the row count months later.
    thin = [(s, n, a, r) for s, n, a, r in season_rates if r < args.min_acceptance]
    print("\n=== fetch summary ===")
    for s, n, a, r in season_rates:
        flag = "  <-- INCOMPLETE" if r < args.min_acceptance else ""
        print(f"  season {s}: accepted {a}/{n} ({r:.1%}){flag}")
    _report_stats()
    if thin:
        names = ", ".join(str(s) for s, *_ in thin)
        print(
            f"\nABORT: season(s) {names} below --min-acceptance "
            f"{args.min_acceptance:.0%}. No files written.\n"
            "This usually means the upstream box-score API does not carry those "
            "seasons (ncaa-api returns 502 before 2023). Source them elsewhere, "
            "or pass --min-acceptance 0 to accept the gap knowingly.",
            file=sys.stderr,
        )
        return 2

    all_rows.sort(key=lambda r: (r["Season"], r["DayNum"]))
    tourney_rows.sort(key=lambda r: (r["Season"], r["DayNum"]))

    # The /basketball-men/d1/ scoreboard still returns D-II and D-III fixtures.
    # Those programmes appear as one-games-out entries, so requiring a realistic
    # D-I schedule cleanly separates the two populations.  The tournament is
    # filtered on the same basis for consistency.
    if args.min_games > 0:
        played: dict[int, int] = {}
        for row in all_rows:
            played[row["WTeamID"]] = played.get(row["WTeamID"], 0) + 1
            played[row["LTeamID"]] = played.get(row["LTeamID"], 0) + 1
        keep = {tid for tid, n in played.items() if n >= args.min_games}
        before = len(all_rows)
        all_rows = [r for r in all_rows if r["WTeamID"] in keep and r["LTeamID"] in keep]
        tourney_rows = [r for r in tourney_rows if r["WTeamID"] in keep and r["LTeamID"] in keep]
        team_names = {tid: nm for tid, nm in team_names.items() if tid in keep}
        print(
            f"min-games>={args.min_games}: kept {len(keep)} teams, "
            f"dropped {before - len(all_rows)} rows"
        )

    def _write(rows: list[dict], path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=GAME_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    reg_path = args.out / "MRegularSeasonDetailedResults.csv"
    tour_path = args.out / "MTourneyDetailedResults.csv"
    _write(all_rows, reg_path)
    _write(tourney_rows, tour_path)

    teams_path = args.out / "MTeams.csv"
    with teams_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["TeamID", "TeamName"])
        for tid in sorted(team_names):
            writer.writerow([tid, team_names[tid]])

    print(f"wrote {len(all_rows)} regular-season games -> {reg_path}")
    print(f"wrote {len(tourney_rows)} tournament games    -> {tour_path}")
    print(f"wrote {len(team_names)} teams                  -> {teams_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
