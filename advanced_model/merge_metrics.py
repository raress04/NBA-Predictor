import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

retro_bias = read_file(os.path.join(MODEL_DIR, 'analyze', 'compute_retroactive_bias.py'))
daily = read_file(os.path.join(MODEL_DIR, 'ml', 'daily_metrics.py'))
shadow = read_file(os.path.join(MODEL_DIR, 'ml', 'evaluate_shadow.py'))

combined = '"""\nperformance_metrics.py\nConsolidates retroactive bias, daily metrics, and shadow pick evaluation.\n"""\n\n'
combined += retro_bias
combined += '\n\n# --- FROM ml/daily_metrics.py ---\n\n'
combined += daily
combined += '\n\n# --- FROM ml/evaluate_shadow.py ---\n\n'
combined += shadow

with open(os.path.join(MODEL_DIR, 'analyze', 'performance_metrics.py'), 'w') as f:
    f.write(combined)

DATE = "2026-08-01"

def create_shim(file_path, new_module_path, imported_items="*"):
    header = f"# DEPRECATED – see {new_module_path}. Kept for backward compatibility until {DATE}.\n"
    if imported_items == "*":
        code = f"{header}from {new_module_path} import *\n"
    else:
        code = f"{header}from {new_module_path} import {imported_items}\n"
    
    with open(file_path, 'w') as f:
        f.write(code)

create_shim(os.path.join(MODEL_DIR, 'analyze', 'compute_retroactive_bias.py'), "analyze.performance_metrics")
create_shim(os.path.join(MODEL_DIR, 'ml', 'daily_metrics.py'), "analyze.performance_metrics")
create_shim(os.path.join(MODEL_DIR, 'ml', 'evaluate_shadow.py'), "analyze.performance_metrics")

# Also update existing shims to the required header format
def update_shim_header(file_path, new_module_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # We strip the first few lines and add our standardized header
    lines = content.split('\n')
    idx = 0
    for i, line in enumerate(lines):
        if line.startswith('from '):
            idx = i
            break
            
    header = f"# DEPRECATED – see {new_module_path}. Kept for backward compatibility until {DATE}.\n"
    new_content = header + '\n'.join(lines[idx:])
    with open(file_path, 'w') as f:
        f.write(new_content)

update_shim_header(os.path.join(MODEL_DIR, 'etl', 'bet_tracker.py'), 'etl.db_manager')
update_shim_header(os.path.join(MODEL_DIR, 'etl', 'live_scraper.py'), 'etl.live_ingestion')
update_shim_header(os.path.join(MODEL_DIR, 'simulator', 'matrix_builder.py'), 'simulator.context_builder')
update_shim_header(os.path.join(MODEL_DIR, 'simulator', 'synergy_tracker.py'), 'simulator.context_builder')
update_shim_header(os.path.join(MODEL_DIR, 'ml', 'build_feature_matrix.py'), 'ml.feature_pipeline')
update_shim_header(os.path.join(MODEL_DIR, 'ml', 'build_training_dataset.py'), 'ml.feature_pipeline')
update_shim_header(os.path.join(MODEL_DIR, 'ml', 'train_calibration_model.py'), 'ml.xgb_engine')
update_shim_header(os.path.join(MODEL_DIR, 'ml', 'train_residual_model.py'), 'ml.xgb_engine')
update_shim_header(os.path.join(MODEL_DIR, 'ml', 'inference.py'), 'ml.feature_pipeline and ml.xgb_engine')

print("All metrics files merged, shims created, and headers updated.")
