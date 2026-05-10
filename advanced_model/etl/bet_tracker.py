# DEPRECATED – see etl.db_manager. Kept for backward compatibility until 2026-08-01.
from etl.db_manager import (
    _get_conn, query_by_type, query_summary, query_confidence_bands,
    ingest_shadow_picks, ingest_shadow_pick, seed_all, seed_march3, seed_march4,
    print_summary, print_lines_summary, check_db, compare_bias
)

if __name__ == '__main__':
    # Forward CLI executions to db_manager
    import sys
    import runpy
    sys.argv[0] = 'advanced_model.etl.db_manager'
    runpy.run_module('advanced_model.etl.db_manager', run_name='__main__')
