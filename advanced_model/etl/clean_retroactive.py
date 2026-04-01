import sqlite3
import os

def clean_old_retroactive():
    # Relative path from project root
    db_path = 'database/bet_tracker.db'
    if not os.path.exists(db_path):
        # Maybe running from inside etl/
        db_path = '../database/bet_tracker.db'
        
    conn = sqlite3.connect(db_path)
    count = conn.execute("""
        SELECT COUNT(*) FROM projection_outcomes 
        WHERE source_mode = 'retroactive_live'
    """).fetchone()[0]
    print(f'Deleting {count} old retroactive_live rows...')
    conn.execute("DELETE FROM projection_outcomes WHERE source_mode = 'retroactive_live'")
    conn.commit()
    conn.close()
    print('Done.')

if __name__ == '__main__':
    clean_old_retroactive()
