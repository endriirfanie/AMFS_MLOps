"""
ground_truth_deploy.py — generated from generic_ml_ground_truth_deploy_pipeline.ipynb.

Runs unattended (via the per-use-case use_case_runner.ipynb, which does
runpy.run_path(this_file, init_globals={"CONFIG_PATH": ...})), or standalone
for interactive debugging (reads "config.yaml" from its own folder by default).
Lives in the shared "AMFS - ML Setup" folder, next to ml_pipeline_utils.py —
never copied per use case.
"""


# ============================================================================
# Cell: g1_config_session
# ============================================================================
import sys, os, json
import numpy as np, pandas as pd
sys.path.append(".")                      # ml_pipeline_utils.py alongside this notebook, or...
from ml_pipeline_utils import *

# CONFIG_PATH: injected by the runner notebook via
# runpy.run_path(..., init_globals={"CONFIG_PATH": ...}) before this script runs.
# Falls back to "config.yaml" (this script's own folder) if run standalone.
CONFIG_PATH = globals().get("CONFIG_PATH", "config.yaml")
CONFIG   = load_config(CONFIG_PATH)
RESOLVED = resolve_names(CONFIG)
session  = get_session(CONFIG)

session.sql(f'USE WAREHOUSE {CONFIG["snowflake"]["warehouse"]}').collect()
session.sql(f'USE SCHEMA {CONFIG["snowflake"]["database"]}.{CONFIG["snowflake"]["schema"]}').collect()

FQ = f'{CONFIG["snowflake"]["database"]}.{CONFIG["snowflake"]["schema"]}'
RT = CONFIG.get("retraining") or {}
assert RT.get("ground_truth_source"), "config.yaml has no retraining.ground_truth_source to deploy a back-fill for"
SYNTHETIC = bool(RT.get("synthetic_labels", False))

print(f'use case: {CONFIG["use_case"]["name"]}  |  task_type: {CONFIG["model"]["task_type"]}')
print(f'outcome source: {FQ}.{RT["ground_truth_source"]}')
print(f'monitor table:  {FQ}.{RESOLVED["monitor_table"]}')
print(f'procedure:      {FQ}.{RESOLVED["gt_procedure_name"]}')
if SYNTHETIC:
    print("!! retraining.synthetic_labels is true: the outcomes are SIMULATED, so schedule this in the sandbox only.")

# ============================================================================
# Cell: g2_check_source
# ============================================================================
gt_sql = RT.get("ground_truth_source_sql_file")
if gt_sql:
    path = os.path.join(CONFIG["_meta"]["config_dir"], gt_sql)
    print(f'creating the outcome source from use-case SQL: {path}')
    run_sql_script(session, path, sql_params(CONFIG, RESOLVED))

CHECK = check_ground_truth_source(session, CONFIG, RESOLVED)
print(f'{RT["ground_truth_source"]}: {CHECK["source_rows"]:,} rows | {CHECK["distinct_keys"]:,} distinct keys | '
      f'{CHECK["null_outcomes"]:,} without an outcome')
print(f'{RESOLVED["monitor_table"]}: {CHECK["monitor_rows"]:,} rows | {CHECK["monitor_pending"]:,} waiting for an outcome | '
      f'{CHECK["labelable_now"]:,} of those the source can label right now')
if CHECK["monitor_pending"] > 0 and CHECK["labelable_now"] == 0:
    print("!! none of the waiting rows match a key in the source. Check the key column and its format, "
          "and (if respect_label_maturity is on) that outcomes exist for these rows.")

# ============================================================================
# Cell: g3_deploy_procedure
# ============================================================================
ddl = deploy_ground_truth_procedure(session, CONFIG, RESOLVED)
print(f'{RESOLVED["gt_procedure_name"]} deployed ({len(ddl):,} chars of generated SQL)')
print(f'maturity rule: {"only rows whose LABEL_MATURE_AT has passed" if RT.get("respect_label_maturity", True) else "label immediately (respect_label_maturity is false)"}')

# ============================================================================
# Cell: g4_dry_run
# ============================================================================
print(session.sql(f'CALL {FQ}.{RESOLVED["gt_procedure_name"]}()').collect()[0][0])

session.sql(f'''
    SELECT COUNT(*) AS TOTAL_ROWS,
           COUNT_IF(ACTUAL_LABEL IS NOT NULL) AS LABELED,
           COUNT_IF(ACTUAL_LABEL IS NULL)     AS WAITING
    FROM {FQ}.{RESOLVED["monitor_table"]}''').show()

# the run just wrote a row to the shared scoring log (most recent first)
session.sql(f'SELECT * FROM {FQ}.{RESOLVED["log_table"]} ORDER BY 1 DESC LIMIT 5').show()

print("Safe to run again: only rows still waiting for an outcome are touched, so a second call should report 0 new rows.")

# ============================================================================
# Cell: g5_airflow_handoff
# ============================================================================
proc_fqn = f'{FQ}.{RESOLVED["gt_procedure_name"]}()'
print("Hand this to the Airflow team:\n")
print(f'  CALL {proc_fqn};')
print()
print("Returns a STRING such as 'OK: labeled 1250 rows, 310 still waiting'. Labeling 0 rows is a normal success.")
print(f'On failure it raises a SQL exception and writes a FAILED row to {RESOLVED["log_table"]}, so it shows up as an ordinary task failure.')
print()
print("Suggested schedule: daily, after the system that produces the outcomes has refreshed. Run it before the retraining task.")
print()
print("One-time grant an admin needs to run so Airflow's service role can call it:")
print(f'  GRANT USAGE ON PROCEDURE {proc_fqn} TO ROLE AIRFLOW_SERVICE_ROLE;')
print()
print("The procedure runs as its owner, so the OWNER role (not Airflow's) needs UPDATE on "
      f'{RESOLVED["monitor_table"]} and SELECT on {RT["ground_truth_source"]}.')
print("The retraining notebook runs the same merge as a safety net, so nothing breaks if this schedule is late.")
print("No Snowflake Tasks were created - scheduling is owned by Airflow's DAG, not this notebook.")
