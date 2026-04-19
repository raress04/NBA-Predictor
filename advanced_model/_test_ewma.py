import sys
sys.path.insert(0, '/mnt/c/Users/balca/OneDrive/Projects/AiBetTool/advanced_model')
from simulator.matrix_builder import get_player_stats_matrix

# Test 1: basic call (no team_id — backward compat)
m = get_player_stats_matrix('LeBron James', limit=15)
print(f"[OK] LeBron — avg_pts: {m['avg_pts']:.1f} | high_uncertainty: {m['_high_uncertainty']} | n_games: {m['_n_games']}")

# Test 2: with wrong team_id (simulates fresh trade, <5 games)
m2 = get_player_stats_matrix('LeBron James', limit=15, current_team_id=999999)
print(f"[OK] LeBron (bad team_id=999999) — avg_pts: {m2['avg_pts']:.1f} | high_uncertainty: {m2['_high_uncertainty']} (expected True)")

# Test 3: Deni Avdija — Portland Trail Blazers team_id = 1610612757
try:
    m3 = get_player_stats_matrix('Deni Avdija', limit=40, current_team_id=1610612757)
    print(f"[OK] Avdija — avg_pts: {m3['avg_pts']:.1f} | high_uncertainty: {m3['_high_uncertainty']} | n_games: {m3['_n_games']}")
    print(f"     Target: avg_pts >= 18.0 — {'PASS' if m3['avg_pts'] >= 18.0 else 'FAIL (still using old data?)'}")
except Exception as e:
    print(f"[ERR] Avdija: {e}")

print("\nAll tests done.")
