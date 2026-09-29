"""
retraining.py — generated from generic_ml_retraining_pipeline.ipynb.

Runs unattended (via the per-use-case use_case_runner.ipynb, which does
runpy.run_path(this_file, init_globals={"CONFIG_PATH": ...})), or standalone
for interactive debugging (reads "config.yaml" from its own folder by default).
Lives in the shared "AMFS - ML Setup" folder, next to ml_pipeline_utils.py —
never copied per use case.
"""


# ============================================================================
# Cell: r1_config_session
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

FQ   = f'{CONFIG["snowflake"]["database"]}.{CONFIG["snowflake"]["schema"]}'
KEYS = CONFIG["data"]["key_columns"]
SEED = CONFIG["split"]["random_seed"]
TASK = CONFIG["model"]["task_type"]          # binary | multiclass | regression

RETRAIN = CONFIG["retraining"]
assert RETRAIN.get("enabled", False), "retraining.enabled is false in config.yaml"
OLD_VER, NEW_VER = resolve_retrain_versions(session, CONFIG, RESOLVED)
SYNTHETIC = bool(RETRAIN.get("synthetic_labels", False))

print(f'use case: {CONFIG["use_case"]["name"]}  |  task_type: {TASK}  |  challenging {OLD_VER} with {NEW_VER}')
if SYNTHETIC:
    print("!! retraining.synthetic_labels is true: outcomes are SIMULATED. This run only proves the pipeline works, "
          "and a version it promotes must not be used for production.")

# ============================================================================
# Cell: r2_ground_truth_merge
# ============================================================================
# Step 1: the use case says where true outcomes live. The merge itself is generic and repeatable.
# If the scheduled ground-truth procedure is running, this is just a safety net and labels 0 rows.
gt_sql = RETRAIN.get("ground_truth_source_sql_file")
if gt_sql:
    path = os.path.join(CONFIG["_meta"]["config_dir"], gt_sql)
    print(f'creating the ground-truth source from use-case SQL: {path}')
    run_sql_script(session, path, sql_params(CONFIG, RESOLVED))

if RETRAIN.get("ground_truth_source"):
    GT = merge_ground_truth(session, CONFIG, RESOLVED)
    print(f'{RESOLVED["monitor_table"]}: {GT["total_rows"]:,} rows | labeled {GT["labeled_before"]:,} -> '
          f'{GT["labeled_after"]:,} (+{GT["newly_labeled"]:,}) | still waiting for an outcome: {GT["still_unlabeled"]:,}')
else:
    print("retraining.ground_truth_source not set - assuming ACTUAL_LABEL is already maintained elsewhere")

# ============================================================================
# Cell: r3_retrain_base_view
# ============================================================================
# Step 2: the dataset the challenger trains on (original training data + labeled production rows).
base_sql = RETRAIN.get("retrain_base_sql_file")
if base_sql:
    path = os.path.join(CONFIG["_meta"]["config_dir"], base_sql)
    print(f'running use-case retrain-base SQL instead of the generic view: {path}')
    run_sql_script(session, path, sql_params(CONFIG, RESOLVED))
else:
    COUNTS = create_retrain_base_view(session, CONFIG, RESOLVED)
    print(f'generic retrain dataset view built: {COUNTS}')

retrain_source = RETRAIN["retrain_base_view"]
n_rows = session.sql(f'SELECT COUNT(*) N FROM {FQ}.{retrain_source}').to_pandas()["N"].iloc[0]
print(f'{retrain_source}: {n_rows:,} rows available for retraining')

# ============================================================================
# Cell: r4_load_champion_and_data
# ============================================================================
CHAMPION = load_past_run(session, CONFIG, RESOLVED, model_version=OLD_VER)
assert CHAMPION is not None, (
    f'no run found for version {OLD_VER} in {RESOLVED["config_table"]} - '
    f'run the training notebook first.'
)
CHAMPION_CFG = CHAMPION["config"]
CHAMPION_CFG.setdefault("model", {}).setdefault("task_type", "binary")   # runs logged before task_type existed were binary
assert CHAMPION_CFG["model"]["task_type"] == TASK, (
    f'config.yaml says task_type={TASK}, but champion {OLD_VER} was trained as '
    f'{CHAMPION_CFG["model"]["task_type"]}.'
)
ALGO          = CHAMPION["algorithm"]
BEST_PARAMS   = CHAMPION["extra"]["best_params"]
N_ROUNDS      = CHAMPION["extra"].get("n_estimators")
IMPUTE_VALUES = CHAMPION["extra"].get("impute_values", {})
TARGET        = CHAMPION_CFG["data"]["target_column"]
CONFIG["model"]["algorithm"] = ALGO  # keep the working config in sync with the champion
print(f'champion {OLD_VER}: algorithm={ALGO} | trained {CHAMPION["created_at"]}')

# a version that already exists would be silently reused by the registry step, so fail before spending compute
if NEW_VER.upper() in registry_versions(session, CONFIG, RESOLVED):
    raise ValueError(f'version {NEW_VER} is already registered. Set retraining.new_version to an unused name, '
                     f'or leave it null to let the notebook pick the next one.')

df = session.table(f'{FQ}.{retrain_source}').to_pandas()
df.columns = [c.upper() for c in df.columns]
df, CLASS_LABELS = prepare_target(df, TARGET, TASK)
if CLASS_LABELS:
    print('target label encoding:', {lab: i for i, lab in enumerate(CLASS_LABELS)})

champion_scored = load_champion_predictions(session, CONFIG, RESOLVED, KEYS[0], model_version=OLD_VER)
print(f'rows scored by {OLD_VER} that now have a real outcome: {len(champion_scored):,}')

# a use-case retrain-base SQL that predates ROW_SOURCE: treat the rows the champion scored as production
if "ROW_SOURCE" not in df.columns:
    df["ROW_SOURCE"] = np.where(df[KEYS[0]].isin(set(champion_scored[KEYS[0]])), "PRODUCTION", "HISTORY")
df = df.reset_index(drop=True)
df["_ROW"] = np.arange(len(df))

# Comparison set: production rows the champion scored without seeing their outcome, newest first.
# Older labeled production rows and all history train the challenger, so it does learn from new data.
champ = champion_scored.rename(columns={c: f"CHAMPION_{c}" for c in champion_scored.columns if c != KEYS[0]})
comp  = df[df["ROW_SOURCE"] == "PRODUCTION"].merge(champ, on=KEYS[0], how="inner")
te    = pick_comparison_holdout(comp, "CHAMPION_SCORED_AT", RETRAIN.get("holdout_fraction", 0.5), SEED)
tr    = df[~df["_ROW"].isin(te["_ROW"])]

min_rows, min_pos = RETRAIN.get("min_holdout_rows", 1000), RETRAIN.get("min_holdout_positives", 20)
SKIP_REASON = None
if len(te) < min_rows:
    SKIP_REASON = (f'only {len(te):,} labeled production rows scored by {OLD_VER} for the comparison '
                   f'(min_holdout_rows is {min_rows:,})')
elif TASK == "binary" and int(te[TARGET].sum()) < min_pos:
    SKIP_REASON = (f'only {int(te[TARGET].sum())} positives in the comparison set '
                   f'(min_holdout_positives is {min_pos})')
elif TASK == "multiclass" and te[TARGET].nunique() < 2:
    SKIP_REASON = 'the comparison set contains a single class'
elif len(tr) == 0:
    SKIP_REASON = 'no rows left to train the challenger on'

if SKIP_REASON:
    print(f'SKIPPED: {SKIP_REASON}')
else:
    X_tr, y_tr = encode_features(tr, CHAMPION_CFG), tr[TARGET]
    FEATURE_ORDER = list(X_tr.columns)
    X_te, y_te = encode_features(te, CHAMPION_CFG, FEATURE_ORDER), te[TARGET]
    X_tr, X_te = X_tr.fillna(IMPUTE_VALUES), X_te.fillna(IMPUTE_VALUES)

    # new data can have gaps the champion never saw: impute them from the training rows and record it
    new_imp = {c: float(0.0 if pd.isna(X_tr[c].mean()) else X_tr[c].mean()) for c in X_tr.columns[X_tr.isna().any()]}
    if new_imp:
        print(f'imputing columns the champion had no gaps in: {list(new_imp)}')
        IMPUTE_VALUES = {**IMPUTE_VALUES, **new_imp}
        X_tr, X_te = X_tr.fillna(new_imp), X_te.fillna(new_imp)
    X_tr, X_te = X_tr.astype("float64"), X_te.astype("float64")
    assert not X_tr.isna().any().any(), "NaNs remain in X_tr after imputation"

    ENC2BASE = build_enc2base_map(FEATURE_ORDER, CHAMPION_CFG["data"]["categorical_columns"])
    print(f'challenger fit = {len(tr):,} rows {tr["ROW_SOURCE"].value_counts().to_dict()} | '
          f'comparison = {len(te):,} labeled production rows | encoded features = {len(FEATURE_ORDER)}')
    if TASK == "binary":
        print(f'base rate: fit {y_tr.mean():.3%} | comparison {y_te.mean():.3%}')

# ============================================================================
# Cell: r5_train_challenger
# ============================================================================
CUT_HI = CUT_MD = THRESHOLD = None
if SKIP_REASON:
    print(f'skipped: {SKIP_REASON}')
else:
    if RETRAIN.get("rerun_hpo", False):
        print("rerun_hpo=true - running a fresh HPO search instead of reusing champion params")
        in_mask = split_hpo(tr, CONFIG)
        X_in, y_in, X_va, y_va = X_tr[in_mask], y_tr[in_mask], X_tr[~in_mask], y_tr[~in_mask]
        BEST_PARAMS, N_ROUNDS, HPO_RESULTS = run_hpo(CONFIG, ALGO, X_in, y_in, X_va, y_va, SEED)
    else:
        print(f'reusing champion {OLD_VER} hyperparameters (retraining.rerun_hpo=false):')
    print(json.dumps(BEST_PARAMS, indent=2, default=str))

    challenger = fit_final_model(CHAMPION_CFG, ALGO, BEST_PARAMS, N_ROUNDS, X_tr, y_tr, SEED)

    # score_new: probability (binary) or confidence (multiclass) | yhat_new: the hard prediction
    if TASK == "regression":
        score_new, yhat_new = None, challenger.predict(X_te)
    elif TASK == "binary":
        score_new = challenger.predict_proba(X_te)[:, 1]
    else:
        score_new, yhat_new = challenger.predict_proba(X_te).max(axis=1), challenger.predict(X_te)

    if TASK in ("binary", "multiclass"):
        CUT_HI, CUT_MD = risk_cutoffs(score_new, CONFIG["scoring"]["risk_band_high_n"],
                                      CONFIG["scoring"]["risk_band_medium_n"])
        print(f'risk band: HIGH >= {CUT_HI:.4f} | MEDIUM >= {CUT_MD:.4f}')
    if TASK == "binary":
        THRESHOLD = CUT_HI
        yhat_new = (score_new >= THRESHOLD).astype(int)

# ============================================================================
# Cell: r6_compare_and_decide
# ============================================================================
PROMOTE, DECISION = False, {}
if SKIP_REASON:
    print(f'skipped: {SKIP_REASON}')
else:
    yv = te[TARGET].values
    if TASK == "regression":
        score_champion = None
        yhat_champion  = te["CHAMPION_PREDICTED_VALUE"].astype(float).values
    else:
        score_champion = te["CHAMPION_PREDICTED_PROBABILITY"].astype(float).values
        yhat_champion  = te["CHAMPION_PREDICTION"].astype(int).values

    champion_metrics   = score_predictions(yv, score_champion, yhat_champion, TASK)
    challenger_metrics = score_predictions(yv, score_new, np.asarray(yhat_new), TASK)

    comparison = pd.DataFrame([{"model": OLD_VER, **champion_metrics},
                               {"model": NEW_VER, **challenger_metrics}]).set_index("model")
    print(f'comparing on {len(yv):,} labeled production rows the champion scored:\n')
    print(comparison.round(4).T.to_string())

    DECISION = evaluate_promotion(champion_metrics, challenger_metrics, RETRAIN["promotion"], TASK)
    print()
    for k, v in DECISION.items():
        if k != "promote":
            print(f'{k:28s} {v:+.4f}')
    PROMOTE = bool(DECISION["promote"])
    print(f'\nDECISION: {"PROMOTE " + NEW_VER if PROMOTE else f"HOLD -- keep {OLD_VER}"}')

    METRICS = dict(challenger_metrics)
    if TASK == "binary":
        METRICS.update({"base_rate": float(yv.mean()), "threshold": float(THRESHOLD)})
    METRICS.update({f"{k}_prev": v for k, v in champion_metrics.items()})
    METRICS.update(DECISION)

# ============================================================================
# Cell: r7_register_if_promoted
# ============================================================================
from snowflake.ml.registry import Registry

if not PROMOTE:
    print(f'{NEW_VER} not registered ({"skipped" if SKIP_REASON else "champion held"}).')
else:
    reg = Registry(session, database_name=CONFIG["snowflake"]["database"], schema_name=CONFIG["snowflake"]["schema"])
    threshold_note = f'threshold={THRESHOLD:.4f}' if THRESHOLD is not None else 'no threshold'
    mv = reg.log_model(
        challenger, model_name=RESOLVED["model_name"], version_name=NEW_VER,
        sample_input_data=X_tr.head(100).astype("float64"),
        metrics=METRICS, options={"enable_explainability": True},
        target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
        comment=(f'Retrained from {len(tr):,} rows, challenging {OLD_VER} | {threshold_note}'
                 + (' | SYNTHETIC LABELS - not for production' if SYNTHETIC else '')),
    )
    print(f'version {NEW_VER} registered')

    RESOLVED_NEW = dict(RESOLVED, model_version=NEW_VER)
    log_feature_contract(session, CHAMPION_CFG, RESOLVED_NEW, FEATURE_ORDER, ENC2BASE)
    log_run(session, CONFIG, RESOLVED_NEW, METRICS, extra={
        "best_params": BEST_PARAMS, "n_estimators": N_ROUNDS, "impute_values": IMPUTE_VALUES,
        "risk_cut_high": CUT_HI, "risk_cut_medium": CUT_MD, "retrained_from": OLD_VER,
        "train_rows": int(len(tr)), "comparison_rows": int(len(te)), "synthetic_labels": SYNTHETIC,
    })
    print(f'{RESOLVED["contract_table"]} and {RESOLVED["config_table"]} updated for {NEW_VER}')

# ============================================================================
# Cell: r8_baseline_v2
# ============================================================================
if not PROMOTE:
    print("no baseline needed: the champion stays.")
else:
    contrib_te = feature_contributions(ALGO, challenger, X_te, FEATURE_ORDER, ENC2BASE)
    A, cols = contrib_te.values, np.array(contrib_te.columns)
    n_causes = CONFIG["explainability"]["top_n_causes_per_row"]
    idx = np.argsort(-A, axis=1)[:, :n_causes]
    causes = [" | ".join(cols[r][A[i, r] > 0]) for i, r in enumerate(idx)]

    BASELINE_V2 = f'{RESOLVED["baseline_table"]}_{NEW_VER}'
    horizon = CONFIG["scoring"]["label_horizon_days"]
    now = pd.Timestamp.now()

    base = pd.concat([te[KEYS].reset_index(drop=True), X_te.reset_index(drop=True)], axis=1)
    if TASK == "regression":
        base["PREDICTED_VALUE"] = yhat_new
        base["PREDICTION"]      = yhat_new
        base["RISK_BAND"]       = None
    elif TASK == "binary":
        base["PREDICTED_PROBABILITY"] = score_new
        base["PREDICTION"]            = np.asarray(yhat_new).astype("int8")
        base["RISK_BAND"]             = risk_band(score_new, CUT_HI, CUT_MD)
    else:  # multiclass
        base["PREDICTED_PROBABILITY"] = score_new
        base["PREDICTION"]            = np.asarray(yhat_new)
        base["RISK_BAND"]             = risk_band(score_new, CUT_HI, CUT_MD)
    base["PROBABLE_CAUSES"] = causes
    base["ACTUAL_LABEL"]    = te[TARGET].values
    base["MODEL_VERSION"]   = NEW_VER
    base["SCORED_AT"]       = now.strftime("%Y-%m-%d %H:%M:%S")
    base["LABEL_MATURE_AT"] = (now + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d %H:%M:%S")
    if TASK == "multiclass":   # Model Monitor needs class columns as text for multiclass models
        base["PREDICTION"]   = base["PREDICTION"].astype(str)
        base["ACTUAL_LABEL"] = base["ACTUAL_LABEL"].astype(str)

    session.sql(f'DROP TABLE IF EXISTS {FQ}.{BASELINE_V2}').collect()
    session.write_pandas(base, BASELINE_V2, database=CONFIG["snowflake"]["database"],
                         schema=CONFIG["snowflake"]["schema"], auto_create_table=True, overwrite=True,
                         quote_identifiers=False)
    session.sql(f'''
    CREATE OR REPLACE TABLE {FQ}.{BASELINE_V2} AS
    SELECT * EXCLUDE (SCORED_AT, LABEL_MATURE_AT),
           SCORED_AT::TIMESTAMP_NTZ(6)       AS SCORED_AT,
           LABEL_MATURE_AT::TIMESTAMP_NTZ(6) AS LABEL_MATURE_AT
    FROM {FQ}.{BASELINE_V2}''').collect()

    mon_cols  = [r[0] for r in session.sql(f'DESC TABLE {FQ}.{RESOLVED["monitor_table"]}').collect()]
    base_cols = [r[0] for r in session.sql(f'DESC TABLE {FQ}.{BASELINE_V2}').collect()]
    print(f'schema parity with {RESOLVED["monitor_table"]}: '
          f'{"OK" if mon_cols == base_cols else "MISMATCH"} ({len(mon_cols)} vs {len(base_cols)})')
    if mon_cols != base_cols:
        print(' only in monitor :', [c for c in mon_cols if c not in base_cols][:10])
        print(' only in baseline:', [c for c in base_cols if c not in mon_cols][:10])
    print(f'{BASELINE_V2}: {len(base):,} rows for {NEW_VER}')

# ============================================================================
# Cell: r9_model_monitor
# ============================================================================
if not PROMOTE:
    print("no model monitor needed: the champion stays.")
elif CONFIG.get("model_monitor", {}).get("enabled", False):
    monitor_name = f'{RESOLVED["model_name"]}_MONITOR_{NEW_VER}'
    create_or_replace_model_monitor(
        session, CONFIG, monitor_name, RESOLVED["model_name"], NEW_VER,
        BASELINE_V2, RESOLVED["monitor_table"], KEYS[0],
    )
    print(f'model monitor {monitor_name} created, watching {NEW_VER} against {BASELINE_V2}')
else:
    print("model_monitor.enabled is false - skipping model monitor creation")

# ============================================================================
# Cell: r10_outcome_and_handoff
# ============================================================================
# Always runs, so an orchestrator can read the outcome from the scoring log instead of parsing notebook output.
STATUS = "SKIPPED" if SKIP_REASON else ("PROMOTED" if PROMOTE else "HELD")
detail = SKIP_REASON or ", ".join(f"{k}={v:+.4f}" for k, v in DECISION.items() if k != "promote")
log_retrain_outcome(session, CONFIG, RESOLVED, OLD_VER, NEW_VER, STATUS,
                    detail + (" [synthetic labels]" if SYNTHETIC else ""))
print(f'{STATUS}: {OLD_VER} -> {NEW_VER}  ({detail})')

if PROMOTE:
    print()
    print("Next: scoring still uses the old version until inference is redeployed.")
    print(f'  1. set inference.deploy_version: "{NEW_VER}" in config.yaml')
    print("  2. re-run generic_ml_inference_deploy_pipeline.ipynb (the new thresholds are baked into the procedure)")
    if SYNTHETIC:
        print("  !! this version was trained on SIMULATED outcomes - sandbox use only.")
