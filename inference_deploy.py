"""
inference_deploy.py — generated from generic_ml_inference_deploy_pipeline.ipynb.

Runs unattended (via the per-use-case use_case_runner.ipynb, which does
runpy.run_path(this_file, init_globals={"CONFIG_PATH": ...})), or standalone
for interactive debugging (reads "config.yaml" from its own folder by default).
Lives in the shared "AMFS - ML Setup" folder, next to ml_pipeline_utils.py —
never copied per use case.
"""


# ============================================================================
# Cell: i1_config_session
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

FQ  = f'{CONFIG["snowflake"]["database"]}.{CONFIG["snowflake"]["schema"]}'
INF = CONFIG.get("inference") or {}
assert INF.get("enabled", False), "config.yaml has no inference: section, or inference.enabled is false"

TASK      = CONFIG["model"]["task_type"]      # binary | multiclass | regression
SCORE_COL = "PREDICTED_VALUE" if TASK == "regression" else "PREDICTED_PROBABILITY"

DEPLOY_VERSION = INF.get("deploy_version") or RESOLVED["model_version"]
print(f'use case: {CONFIG["use_case"]["name"]}  |  task_type: {TASK}  |  deploying version: {DEPLOY_VERSION}')
print(f'input view: {FQ}.{RESOLVED["input_view"]}')
print(f'procedure:  {FQ}.{RESOLVED["procedure_name"]}')

# ============================================================================
# Cell: i2_load_target_version
# ============================================================================
RUN = load_past_run(session, CONFIG, RESOLVED, model_version=DEPLOY_VERSION)
assert RUN is not None, (
    f'no run found for version {DEPLOY_VERSION} in {RESOLVED["config_table"]} - '
    f'run training or retraining for that version first.'
)
RUN_CFG = RUN["config"]

# runs logged before task_type existed were all binary
run_task = RUN_CFG.get("model", {}).get("task_type", "binary")
assert run_task == TASK, (
    f'config.yaml says task_type={TASK}, but {DEPLOY_VERSION} was trained as {run_task} - '
    f'deploying it with the wrong task type would write the wrong columns.'
)

CONTRACT = load_feature_contract(session, CONFIG, RESOLVED, DEPLOY_VERSION)
FEATURE_ORDER = list(CONTRACT["ENCODED_NAME"])

# risk-band cutoffs exist for binary + multiclass; regression has none (stored as null)
CUT_HI = RUN["extra"].get("risk_cut_high")
CUT_MD = RUN["extra"].get("risk_cut_medium")
# only binary scores against a probability threshold (training sets it equal to CUT_HI);
# multiclass predicts the argmax class and regression predicts a value
THRESHOLD = RUN["metrics"].get("threshold", CUT_HI) if TASK == "binary" else None
HORIZON_DAYS = RUN_CFG["scoring"]["label_horizon_days"]
IMPUTE_VALUES = RUN["extra"].get("impute_values", {})

print(f'{DEPLOY_VERSION}: algorithm={RUN["algorithm"]} | trained {RUN["created_at"]} | '
      f'{len(FEATURE_ORDER)} encoded features')
if TASK == "binary":
    print(f'threshold={THRESHOLD:.4f} | HIGH>={CUT_HI:.4f} | MEDIUM>={CUT_MD:.4f} | '
          f'label horizon={HORIZON_DAYS}d')
elif TASK == "multiclass":
    print(f'prediction = most likely class | confidence bands: HIGH>={CUT_HI:.4f} | '
          f'MEDIUM>={CUT_MD:.4f} | label horizon={HORIZON_DAYS}d')
else:
    print(f'regression: predicted value only, no threshold or risk bands | '
          f'label horizon={HORIZON_DAYS}d')

parity = check_monitor_feature_parity(session, CONFIG, RESOLVED, FEATURE_ORDER)
if not parity["ok"]:
    print(f'!! {RESOLVED["monitor_table"]} has {parity["table_feature_count"]} feature columns, '
          f'{DEPLOY_VERSION} expects {parity["model_feature_count"]}')
    print('   extra in table  :', parity["extra"][:10])
    print('   missing in table:', parity["missing"][:10])
    raise ValueError(
        'monitor table columns do not match this model version\'s feature contract - '
        'this usually means a retrain changed the feature set. Re-run this notebook only '
        'after confirming that is expected (a schema migration on the monitor table may be needed).'
    )
print(f'feature parity with {RESOLVED["monitor_table"]}: OK ({parity["table_feature_count"]})')

# ============================================================================
# Cell: i3_deploy_input_view
# ============================================================================
cols = deploy_scoring_input_view(
    session, CONFIG, RESOLVED, CONTRACT,
    source_view=INF["source_stream_view"],
    passthrough_cols=INF["passthrough_columns"],
    impute_values=IMPUTE_VALUES,
)
print(f'{RESOLVED["input_view"]}: {len(cols)} columns '
      f'({len(INF["passthrough_columns"])} passthrough + {len(FEATURE_ORDER)} features)')
session.sql(f'SELECT COUNT(*) AS N FROM {FQ}.{RESOLVED["input_view"]}').show()

# ============================================================================
# Cell: i4_probe_and_deploy_procedure
# ============================================================================
# binary: the positive-class key | multiclass: one key per class | regression: the single value key
OUTPUT_KEYS = probe_model_output_keys(
    session, CONFIG, RESOLVED, RESOLVED["model_name"], DEPLOY_VERSION, FEATURE_ORDER)
print(f'{"PREDICT" if TASK == "regression" else "PREDICT_PROBA"} output key(s): {OUTPUT_KEYS}')

HAS_EXPLAIN, EXPL_SUFFIX = probe_explain_suffix(
    session, CONFIG, RESOLVED, RESOLVED["model_name"], DEPLOY_VERSION, FEATURE_ORDER)
print(f'EXPLAIN available: {HAS_EXPLAIN}' + (f' | suffix={EXPL_SUFFIX!r}' if HAS_EXPLAIN else
      ' - PROBABLE_CAUSES will be NULL until it is'))

ddl = deploy_scoring_procedure(
    session, CONFIG, RESOLVED,
    model_name=RESOLVED["model_name"], model_version=DEPLOY_VERSION,
    feature_order=FEATURE_ORDER, proba_key=OUTPUT_KEYS[0], output_keys=OUTPUT_KEYS,
    has_explain=HAS_EXPLAIN, expl_suffix=EXPL_SUFFIX,
    threshold=THRESHOLD, cut_hi=CUT_HI, cut_md=CUT_MD, horizon_days=HORIZON_DAYS,
    watermark_col=INF["watermark_column"], xai_bucket_count=INF["xai_bucket_count"],
)
print(f'{RESOLVED["procedure_name"]} deployed ({len(ddl):,} chars of generated SQL)')

# ============================================================================
# Cell: i5_dry_run
# ============================================================================
print(session.sql(f'CALL {FQ}.{RESOLVED["procedure_name"]}()').collect()[0][0])

if TASK == "regression":
    summary_sql = f'''
        SELECT SCORED_AT::DATE AS DT, MODEL_VERSION, COUNT(*) AS N,
               ROUND(AVG(PREDICTED_VALUE), 4) AS AVG_PRED,
               ROUND(MIN(PREDICTED_VALUE), 4) AS MIN_PRED,
               ROUND(MAX(PREDICTED_VALUE), 4) AS MAX_PRED,
               COUNT(PROBABLE_CAUSES) AS N_WITH_CAUSES,
               COUNT(ACTUAL_LABEL) AS N_ACTUAL
        FROM {FQ}.{RESOLVED["monitor_table"]} GROUP BY 1, 2 ORDER BY 1 DESC LIMIT 10'''
elif TASK == "multiclass":
    summary_sql = f'''
        SELECT SCORED_AT::DATE AS DT, MODEL_VERSION, COUNT(*) AS N,
               ROUND(AVG(PREDICTED_PROBABILITY), 4) AS AVG_CONFIDENCE,
               COUNT(DISTINCT PREDICTION) AS N_CLASSES_PREDICTED,
               SUM(IFF(RISK_BAND = 'HIGH', 1, 0)) AS N_HIGH,
               COUNT(PROBABLE_CAUSES) AS N_WITH_CAUSES,
               COUNT(ACTUAL_LABEL) AS N_ACTUAL
        FROM {FQ}.{RESOLVED["monitor_table"]} GROUP BY 1, 2 ORDER BY 1 DESC LIMIT 10'''
else:  # binary
    summary_sql = f'''
        SELECT SCORED_AT::DATE AS DT, MODEL_VERSION, COUNT(*) AS N,
               ROUND(AVG(PREDICTED_PROBABILITY), 4) AS AVG_PROB,
               SUM(PREDICTION) AS N_FLAGGED,
               SUM(IFF(RISK_BAND = 'HIGH', 1, 0)) AS N_HIGH,
               COUNT(PROBABLE_CAUSES) AS N_WITH_CAUSES,
               COUNT(ACTUAL_LABEL) AS N_ACTUAL
        FROM {FQ}.{RESOLVED["monitor_table"]} GROUP BY 1, 2 ORDER BY 1 DESC LIMIT 10'''
session.sql(summary_sql).show()

biz_table = CONFIG["inference"].get("business_table")
if biz_table:
    session.sql(f'''
        SELECT * FROM {FQ}.{biz_table}
        ORDER BY {SCORE_COL} DESC LIMIT 5''').show()

# Drift sanity check: latest batch vs the training-time baseline
session.sql(f'''
    SELECT 'baseline' AS SRC, ROUND(AVG({SCORE_COL}), 4) AS AVG_SCORE, COUNT(*) AS N
    FROM {FQ}.{RESOLVED["baseline_table"]}
    UNION ALL
    SELECT 'latest batch', ROUND(AVG({SCORE_COL}), 4), COUNT(*)
    FROM {FQ}.{RESOLVED["monitor_table"]}
    WHERE SCORED_AT = (SELECT MAX(SCORED_AT) FROM {FQ}.{RESOLVED["monitor_table"]})''').show()

if TASK == "multiclass":
    # the average confidence above hides a shift in *which* class is predicted, so compare the mix too
    session.sql(f'''
        SELECT 'baseline' AS SRC, PREDICTION, COUNT(*) AS N
        FROM {FQ}.{RESOLVED["baseline_table"]} GROUP BY 2
        UNION ALL
        SELECT 'latest batch', PREDICTION, COUNT(*)
        FROM {FQ}.{RESOLVED["monitor_table"]}
        WHERE SCORED_AT = (SELECT MAX(SCORED_AT) FROM {FQ}.{RESOLVED["monitor_table"]}) GROUP BY 2
        ORDER BY 1, 2''').show()

session.sql(f'SELECT * FROM {FQ}.{RESOLVED["watermark_table"]} '
           f'WHERE USE_CASE_NAME = \'{CONFIG["use_case"]["name"]}\'').show()

print("Rollback (dry-run) commands, if this deploy needs to be undone:")
print(f'  DELETE FROM {FQ}.{RESOLVED["monitor_table"]} WHERE ACTUAL_LABEL IS NULL '
      f'AND MODEL_VERSION = \'{DEPLOY_VERSION}\';')
if biz_table:
    print(f'  -- {biz_table} is upserted, not append-only - restoring it needs a backup, not a DELETE')
print(f'  UPDATE {FQ}.{RESOLVED["watermark_table"]} SET LAST_RECORD_TS = \'2000-01-01\' '
      f'WHERE USE_CASE_NAME = \'{CONFIG["use_case"]["name"]}\';')

# ============================================================================
# Cell: i6_airflow_handoff
# ============================================================================
proc_fqn = f'{FQ}.{RESOLVED["procedure_name"]}()'
print("Hand this to the Airflow team (also written to airflow_trigger.sql):\n")
print(f'  CALL {proc_fqn};')
print()
print("Returns a STRING: 'OK: N rows scored' or 'SKIPPED: 0 new rows' (both success).")
print("On failure it raises a SQL exception after rolling back and logging to")
print(f'{RESOLVED["log_table"]} - a normal Airflow task failure, no partial writes to check for.')
print()
print("One-time grant an admin needs to run so Airflow's service role can call it:")
print(f'  GRANT USAGE ON PROCEDURE {proc_fqn} TO ROLE AIRFLOW_SERVICE_ROLE;')
print()
print("No Snowflake Tasks were created - scheduling is owned by Airflow's DAG, not this notebook.")
