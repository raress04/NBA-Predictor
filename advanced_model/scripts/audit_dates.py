"""
Read-only date-consistency audit (06.11.2026).

Ground truth = nba_data.db (games + box_scores, NBA-API dates = US-Eastern game dates).
Checks every dated prediction / results / analyze file (txt + pdf) and every dated
table in bet_tracker.db, nba_data.db and master_training_data.db.

Usage (from AiBetTool/):  python3 tmp/audit_dates.py advanced_model  > report.txt
Never writes to any database or project file (databases opened read-only).
"""
import os
import re
import sqlite3
import subprocess
import sys
import unicodedata
from collections import defaultdict, Counter
from datetime import date, timedelta

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'advanced_model')
DB = os.path.join(ROOT, 'database')
MONTHS = {m: i for i, m in enumerate(
    ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
     'september', 'october', 'november', 'december'], 1)}
ALT_ABBR = {'GS': 'GSW', 'NY': 'NYK', 'NO': 'NOP', 'SA': 'SAS', 'UTAH': 'UTA', 'WSH': 'WAS',
            'PHO': 'PHX', 'BRK': 'BKN', 'CHO': 'CHA', 'NOR': 'NOP', 'GOS': 'GSW'}


def ro(path):
    return sqlite3.connect(f'file:{path}?mode=ro', uri=True)


def norm(name):
    s = unicodedata.normalize('NFKD', str(name or ''))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-zA-Z0-9 ]", ' ', s).lower()
    return ' '.join(p for p in s.split() if p not in {'jr', 'sr', 'ii', 'iii', 'iv', 'v'})


def d(s):
    s = str(s)[:10]
    if re.fullmatch(r'\d{8}', s):
        s = f'{s[:4]}-{s[4:6]}-{s[6:8]}'
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


# ── ground truth ─────────────────────────────────────────────────────────────
nba = ro(os.path.join(DB, 'nba_data.db'))
TEAMS = {}       # abbr -> nickname
for tid, ab, city, nick in nba.execute("SELECT team_id, abbreviation, city, nickname FROM teams"):
    TEAMS[ab] = nick
TID = {tid: ab for tid, ab, _, _ in nba.execute("SELECT team_id, abbreviation, city, nickname FROM teams")}
NICK_RE = [(ab, re.compile(r'\b' + re.escape(n) + r'\b')) for ab, n in TEAMS.items()]
ABBR_RE = re.compile(r'\b(' + '|'.join(sorted(list(TEAMS) + list(ALT_ABBR), key=len, reverse=True)) + r')\b')

game_date = {}
for gid, gd in nba.execute("SELECT game_id, game_date FROM games"):
    game_date[gid] = d(gd)
game_teams = defaultdict(set)
team_pts = defaultdict(int)
player_dates = defaultdict(set)
box_game_dates = defaultdict(set)
seen = set()
for gid, tid, pid, pname, gd, pts, mins in nba.execute(
        "SELECT game_id, team_id, player_id, player_name, game_date, pts, minutes FROM box_scores"):
    box_game_dates[gid].add(d(gd))
    if (gid, pid) in seen:
        continue
    seen.add((gid, pid))
    if tid in TID:
        game_teams[gid].add(TID[tid])
        team_pts[(gid, TID[tid])] += pts or 0
    try:
        played = float(mins or 0) > 0 if ':' not in str(mins) else True
    except ValueError:
        played = True
    if played:
        player_dates[norm(pname)].add(d(gd))

pair_dates = defaultdict(set)
pair_games = defaultdict(list)
for gid, teams in game_teams.items():
    if len(teams) == 2 and game_date.get(gid):
        pair_dates[frozenset(teams)].add(game_date[gid])
        pair_games[(frozenset(teams), game_date[gid])].append(gid)
GT_MIN = min(x for x in game_date.values() if x)
GT_MAX = max(x for x in game_date.values() if x)
DATES_WITH_GAMES = {x for x in game_date.values() if x}


def teams_in(text):
    # bookmaker names contain team nicknames ("Draft Kings" -> Kings)
    text = re.sub(r'Draft\s*Kings|DraftKings', ' ', text, flags=re.I)
    found = set()
    for ab, rx in NICK_RE:
        if rx.search(text):
            found.add(ab)
    if len(found) < 2:
        for m in ABBR_RE.findall(text):
            found.add(ALT_ABBR.get(m, m))
    return found


MATCH_LINE = re.compile(r'(MATCHUP|\s@\s|\svs\.?\s)')


def pairs_in_text(text):
    pairs = Counter()
    for line in text.splitlines():
        if not MATCH_LINE.search(line):
            continue
        core = line.split('|')[0] if '|' in line and 'MATCHUP' not in line else line
        t = teams_in(core)
        if len(t) == 2:
            pairs[frozenset(t)] += 1
    return pairs


def classify_pairs(label, pairs):
    """Return (status, detail)."""
    if not pairs:
        return 'NO_MATCHUPS', ''
    if label < GT_MIN or label > GT_MAX:
        return 'NO_GROUND_TRUTH', f'label outside games table ({GT_MIN}..{GT_MAX})'
    on, off = [], []
    for p in pairs:
        ds = pair_dates.get(p, set())
        if label in ds:
            on.append(p)
        else:
            near = sorted((x for x in ds if abs((x - label).days) <= 7), key=lambda x: abs((x - label).days))
            off.append((p, near[0] if near else None))
    if not off:
        return 'OK', f'{len(on)} games'
    names = lambda p: '/'.join(sorted(p))
    detail = '; '.join(f"{names(p)} played {x or 'not within ±7d'}" for p, x in off)
    if not on:
        shifts = Counter((x - label).days for _, x in off if x)
        if shifts and shifts.most_common(1)[0][1] == len(off):
            return f'SHIFTED({shifts.most_common(1)[0][0]:+d}d)', detail
        return 'ALL_OFF', detail
    return f'MIXED({len(on)} ok/{len(off)} off)', detail


def header_dates(text, n_lines=12):
    out = set()
    head = '\n'.join(text.splitlines()[:n_lines])
    for m in re.findall(r'\b(20\d\d-\d\d-\d\d)\b', head):
        out.add(d(m))
    for mon, dd, yy in re.findall(r'\b(January|February|March|April|May|June|July|August|September|'
                                  r'October|November|December) (\d{1,2}), (20\d\d)\b', head):
        out.add(date(int(yy), MONTHS[mon.lower()], int(dd)))
    return out


def read_text(path):
    if path.endswith('.pdf'):
        try:
            return subprocess.run(['pdftotext', '-layout', path, '-'], capture_output=True,
                                  text=True, timeout=60).stdout
        except Exception as e:
            return f'__PDF_ERROR__ {e}'
    with open(path, 'rb') as f:
        raw = f.read()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return raw.decode('utf-16', 'replace')
    return raw.decode('utf-8', 'replace')


# ── 1. files ─────────────────────────────────────────────────────────────────
FILE_RE = re.compile(r'(predicts(?:_report)?|shadow_predicts|live_predicts|backtest_predicts|'
                     r'yesterday_results|results|analyze(?:_report|_ml_shadow)?)_(20\d\d-\d\d-\d\d)\.(txt|pdf)$')
rows = []
for base in ('predictions', 'results', 'analyze', 'preserved_runs'):
    for dirpath, _, files in os.walk(os.path.join(ROOT, base)):
        for fn in sorted(files):
            m = FILE_RE.search(fn)
            if not m:
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ROOT)
            label = d(m.group(2))
            text = read_text(path)
            pairs = pairs_in_text(text)
            status, detail = classify_pairs(label, pairs)
            # folder month vs label month
            parts = rel.replace('\\', '/').split('/')
            month_dirs = [p for p in parts if p in MONTHS]
            month_ok = (not month_dirs) or MONTHS[month_dirs[-1]] == label.month
            hd = header_dates(text)
            hd_bad = sorted(x for x in hd if x and x != label)
            extra = []
            if not month_ok:
                extra.append(f'folder month {month_dirs[-1]} != {label:%B}')
            if hd_bad:
                extra.append('header dates ' + ','.join(map(str, hd_bad)))
            if label not in DATES_WITH_GAMES and GT_MIN <= label <= GT_MAX:
                extra.append('no NBA games on this date in games table')
            # results score check
            score_bad = 0
            if 'results' in fn:
                for line in text.splitlines():
                    mm = re.match(r'\s*MATCHUP:\s*(\w+)\s+(?:vs\.|@)\s+(\w+)\s+(\d+)\s*-\s*(\d+)', line)
                    if not mm:
                        continue
                    a, b = ALT_ABBR.get(mm.group(1), mm.group(1)), ALT_ABBR.get(mm.group(2), mm.group(2))
                    gids = pair_games.get((frozenset({a, b}), label), [])
                    if gids:
                        g = gids[0]
                        if {int(mm.group(3)), int(mm.group(4))} != {team_pts[(g, a)], team_pts[(g, b)]}:
                            score_bad += 1
                if score_bad:
                    extra.append(f'{score_bad} scores differ from box scores')
            rows.append((rel, label, status, detail, '; '.join(extra)))

print('=' * 100)
print(f'DATE CONSISTENCY AUDIT — ground truth: games {GT_MIN}..{GT_MAX}, {len(game_date)} games')
print('=' * 100)
print('\n## 1. FILES (label date = date in filename)\n')
by_status = Counter(r[2].split('(')[0] for r in rows)
print('Summary by status:', dict(by_status))
by_dir_status = defaultdict(Counter)
for rel, label, status, detail, extra in rows:
    top = '/'.join(rel.replace('\\', '/').split('/')[:3])
    by_dir_status[top][status.split('(')[0] + ('+NOTE' if extra else '')] += 1
for k in sorted(by_dir_status):
    print(f'  {k:60s} {dict(by_dir_status[k])}')
print('\nFiles that are NOT clean (status != OK or has notes):')
for rel, label, status, detail, extra in rows:
    if status != 'OK' or extra:
        print(f'  [{status}] {rel}' + (f'\n      matchups: {detail}' if detail and status != 'OK' else '')
              + (f'\n      notes: {extra}' if extra else ''))

# predictions vs results/analyze coverage per mode (2026-27)
print('\n## 1b. 2026-27 file set per date (P=prediction txt, R=results, A=analyze pdf, L=legacy flat results)\n')
cov = defaultdict(lambda: defaultdict(set))
for rel, label, *_ in rows:
    r = rel.replace('\\', '/')
    if '2026-2027' not in r and not r.startswith('results/october') and not r.startswith('results/november'):
        continue
    mode = 'live' if 'live_' in r else 'backtest' if 'backtest_' in r else 'ml' if 'ml_shadow' in r else 'legacy'
    kind = 'P' if ('predictions/' in r and r.endswith('.txt')) else 'R' if 'results' in r else 'A' if 'analyze' in r else '?'
    if mode == 'legacy':
        kind, mode = 'L', 'live'
    cov[label][mode].add(kind)
for label in sorted(cov):
    print(f'  {label}  live={"".join(sorted(cov[label]["live"])) or "-":5s} backtest={"".join(sorted(cov[label]["backtest"])) or "-":5s} ml={"".join(sorted(cov[label]["ml"])) or "-"}')

# legacy flat results vs season tree
print('\n## 1c. Legacy flat results/<month>/ vs results/2026-2027/<mode>_results/<month>/\n')
for month in ('october', 'november'):
    flat = os.path.join(ROOT, 'results', month)
    if not os.path.isdir(flat):
        continue
    for fn in sorted(os.listdir(flat)):
        a = os.path.join(flat, fn)
        twins = [os.path.join(ROOT, 'results', '2026-2027', f'{m}_results', month, fn) for m in ('live', 'backtest')]
        same = [os.path.relpath(t, ROOT) for t in twins if os.path.exists(t) and open(t, 'rb').read() == open(a, 'rb').read()]
        diff = [os.path.relpath(t, ROOT) for t in twins if os.path.exists(t) and t not in [os.path.join(ROOT, s) for s in same]]
        print(f'  results/{month}/{fn}: identical to {same or "none"}' + (f'; DIFFERS from {diff}' if diff else '')
              + ('' if same or diff else '  <-- ONLY in legacy folder'))

# ── 2. nba_data.db internal ──────────────────────────────────────────────────
print('\n## 2. nba_data.db internal consistency\n')
mism = [g for g, ds in box_game_dates.items() if game_date.get(g) and (len(ds) != 1 or game_date[g] not in ds)]
print(f'  box_scores.game_date != games.game_date: {len(mism)} games')
print(f'  box_scores game_ids missing from games: {sum(1 for g in box_game_dates if g not in game_date)}')
print(f'  games without box scores: {sum(1 for g in game_date if g not in box_game_dates)}')
print(f'  games whose box scores do not show exactly 2 teams: {sum(1 for g, t in game_teams.items() if len(t) != 2)}')
bad_season = 0
for gid, s, gd in nba.execute("SELECT game_id, season, game_date FROM games"):
    x = d(gd)
    if x:
        exp = f'{x.year}-{str(x.year + 1)[2:]}' if x.month >= 9 else f'{x.year - 1}-{str(x.year)[2:]}'
        if s != exp:
            bad_season += 1
print(f'  games with season label inconsistent with date: {bad_season}')
dup = nba.execute("SELECT COUNT(*) FROM (SELECT 1 FROM box_scores GROUP BY game_id, player_id HAVING COUNT(*)>1)").fetchone()[0]
print(f'  duplicate (game_id, player_id) rows: {dup}')
for t, c in (('games', 'game_date'), ('box_scores', 'game_date'), ('historical_injuries', 'game_date')):
    fm = nba.execute(f"SELECT length({c}), COUNT(*) FROM {t} GROUP BY 1").fetchall()
    print(f'  {t}.{c} formats (length: rows): {fm}')
inj = Counter()
for gd, in nba.execute("SELECT DISTINCT substr(game_date,1,10) FROM historical_injuries"):
    x = d(gd)
    if x and GT_MIN <= x <= GT_MAX:
        inj['date has games' if x in DATES_WITH_GAMES else 'date WITHOUT games'] += 1
print(f'  historical_injuries distinct dates: {dict(inj)}')


# ── 3. player/date tables ────────────────────────────────────────────────────
def player_table(conn, dbname, table, pcol, dcol, where='1=1'):
    try:
        data = conn.execute(f"SELECT {dcol}, {pcol} FROM {table} WHERE {where}").fetchall()
    except sqlite3.Error as e:
        print(f'  {dbname}.{table}: ERROR {e}')
        return
    fm = Counter(len(str(x)) if x is not None else 0 for x, _ in data)
    by_date = defaultdict(set)
    for dd_, p in data:
        x = d(dd_)
        if x:
            by_date[x].add(norm(p))
    flagged, ok, nogt = [], 0, 0
    for x in sorted(by_date):
        ps = by_date[x]
        if x < GT_MIN or x > GT_MAX:
            nogt += 1
            continue
        cov = lambda day: sum(1 for p in ps if day in player_dates.get(p, ())) / len(ps)
        c0, cm, cp = cov(x), cov(x - timedelta(days=1)), cov(x + timedelta(days=1))
        if c0 >= 0.6:
            ok += 1
        else:
            tag = 'SHIFT -1d?' if cm > c0 + 0.3 else 'SHIFT +1d?' if cp > c0 + 0.3 else 'LOW'
            flagged.append(f'{x} n={len(ps)} played: same day {c0:.0%}, day-1 {cm:.0%}, day+1 {cp:.0%} [{tag}]')
    print(f'  {dbname}.{table} [{where}] — {len(data)} rows, {len(by_date)} dates; date formats {dict(fm)}; '
          f'{ok} dates OK, {len(flagged)} flagged, {nogt} outside ground truth')
    for f in flagged:
        print(f'      {f}')


print('\n## 3. Player-level tables: do the players have a box score on the stored date?\n'
      '   (OK = >=60% of distinct players played that day; props can exist for players who end up DNP)\n')
bet = ro(os.path.join(DB, 'bet_tracker.db'))
PROPS = "stat_category IN ('POINTS','REBOUNDS','ASSISTS')"
player_table(bet, 'bet_tracker', 'daily_picks', 'player_or_team', 'date', PROPS)
player_table(bet, 'bet_tracker', 'shadow_picks', 'player', 'game_date', PROPS + " AND COALESCE(superseded,0)=0")
player_table(bet, 'bet_tracker', 'ml_shadow_picks', 'player', 'game_date')
player_table(bet, 'bet_tracker', 'projection_outcomes', 'player', 'game_date', "category IN ('POINTS','REBOUNDS','ASSISTS')")
player_table(bet, 'bet_tracker', 'projection_outcomes_raw', 'player', 'game_date', "category IN ('POINTS','REBOUNDS','ASSISTS')")
player_table(bet, 'bet_tracker', 'historical_odds_cache', 'player_name', 'date', PROPS)
mtd = ro(os.path.join(DB, 'master_training_data.db'))
player_table(mtd, 'master_training_data', 'projection_outcomes', 'player', 'game_date', "category IN ('POINTS','REBOUNDS','ASSISTS')")


# ── 4. team-level tables ─────────────────────────────────────────────────────
def team_table(conn, dbname, table, tcol, dcol, where='1=1'):
    try:
        data = conn.execute(f"SELECT {dcol}, {tcol} FROM {table} WHERE {where}").fetchall()
    except sqlite3.Error as e:
        print(f'  {dbname}.{table}: ERROR {e}')
        return
    res = Counter()
    bad = defaultdict(Counter)
    for dd_, txt in data:
        x = d(dd_)
        t = teams_in(str(txt or ''))
        if not x or len(t) != 2:
            res['unparsed'] += 1
            continue
        if x < GT_MIN or x > GT_MAX:
            res['no ground truth'] += 1
            continue
        ds = pair_dates.get(frozenset(t), set())
        if x in ds:
            res['ok'] += 1
        else:
            near = sorted((y for y in ds if abs((y - x).days) <= 7), key=lambda y: abs((y - x).days))
            k = f'{(near[0] - x).days:+d}d' if near else 'no game ±7d'
            res['off ' + k] += 1
            bad[x][k] += 1
    print(f'  {dbname}.{table}.{tcol} [{where}] — {dict(res)}')
    for x in sorted(bad):
        print(f'      {x}: {dict(bad[x])}')


print('\n## 4. Team-level rows: does the matchup exist on the stored date?\n')
team_table(bet, 'bet_tracker', 'daily_picks', 'game', 'date')
team_table(bet, 'bet_tracker', 'historical_odds_cache', 'player_name', 'date', "stat_category = 'TOTAL'")
team_table(bet, 'bet_tracker', 'shadow_picks', 'matchup', 'game_date', "matchup IS NOT NULL")
gr = bet.execute("SELECT date, home_team, away_team, home_score, away_score FROM game_results").fetchall()
grc = Counter()
for dd_, h, a, hs, as_ in gr:
    x = d(dd_)
    g = pair_games.get((frozenset({ALT_ABBR.get(h, h), ALT_ABBR.get(a, a)}), x))
    if not g:
        grc['matchup not on date'] += 1
    elif {hs, as_} != {team_pts[(g[0], ALT_ABBR.get(h, h))], team_pts[(g[0], ALT_ABBR.get(a, a))]}:
        grc['score differs'] += 1
    else:
        grc['ok'] += 1
print(f'  bet_tracker.game_results — {dict(grc)}')

# ── 5. run-time vs stored slate date ─────────────────────────────────────────
print('\n## 5. Run time (created_at, UTC -> US-Eastern date) vs stored game date\n')
# US-Eastern date with DST (was a fixed -5 h, wrong Mar-Nov) — same rule as utils/slate_date.py
sys.path.insert(0, ROOT)
from utils.slate_date import register_sqlite
register_sqlite(bet)
for t in ('shadow_picks', 'ml_shadow_picks'):
    r = bet.execute(f"""SELECT game_date, eastern_date(created_at), COUNT(*) FROM {t}
                        WHERE created_at IS NOT NULL AND game_date >= '2026-10-01'
                        GROUP BY 1,2 HAVING game_date != eastern_date(created_at)""").fetchall()
    print(f'  {t}: batches where stored date != run ET date: {r or "none"}')

# ── 6. date-format consistency in bet_tracker ────────────────────────────────
print('\n## 6. Date formats per column (length: rows) — anything other than 10 (YYYY-MM-DD) is inconsistent\n')
for t, c in (('daily_picks', 'date'), ('shadow_picks', 'game_date'), ('ml_shadow_picks', 'game_date'),
             ('projection_outcomes', 'game_date'), ('projection_outcomes_raw', 'game_date'),
             ('parlays', 'game_date'), ('game_results', 'date'), ('historical_odds_cache', 'date')):
    print(f'  bet_tracker.{t}.{c}: {bet.execute(f"SELECT length({c}), COUNT(*) FROM {t} GROUP BY 1").fetchall()}')
print(f'  master_training_data.projection_outcomes.game_date: '
      f'{mtd.execute("SELECT length(game_date), COUNT(*) FROM projection_outcomes GROUP BY 1").fetchall()}')
print('\nEND')
