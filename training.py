"""
training.py — generated from generic_ml_training_pipeline.ipynb.

Runs unattended (via the per-use-case use_case_runner.ipynb, which does
runpy.run_path(this_file, init_globals={"CONFIG_PATH": ...})), or standalone
for interactive debugging (reads "config.yaml" from its own folder by default).
Lives in the shared "AMFS - ML Setup" folder, next to ml_pipeline_utils.py —
never copied per use case.
"""


# ============================================================================
# Cell: c1_config_session
# ============================================================================
import sys, json
import numpy as np, pandas as pd
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

FQ       = f'{CONFIG["snowflake"]["database"]}.{CONFIG["snowflake"]["schema"]}'
TARGET   = CONFIG["data"]["target_column"]
TS_COL   = CONFIG["data"]["timestamp_column"]
KEYS     = CONFIG["data"]["key_columns"]
CAT_COLS = CONFIG["data"]["categorical_columns"]
NUM_COLS = CONFIG["data"]["numeric_columns"]
FEATURES = CAT_COLS + NUM_COLS
ALGO     = CONFIG["model"]["algorithm"]
SEED     = CONFIG["split"]["random_seed"]
TASK     = CONFIG["model"]["task_type"]          # binary | multiclass | regression

# fail fast on an invalid task_type / optimization_metric combination rather
# than surfacing a cryptic error several cells downstream
VALID_METRICS = {
    "binary": {"pr_auc", "roc_auc", "f1"},
    "multiclass": {"f1"},
    "regression": {"rmse", "mae", "r2"},
}
opt_metric = CONFIG["model"]["optimization_metric"]
if TASK not in VALID_METRICS:
    raise ValueError(f'model.task_type "{TASK}" is invalid. Valid options: {list(VALID_METRICS)}')
if opt_metric not in VALID_METRICS[TASK]:
    raise ValueError(f'optimization_metric "{opt_metric}" is not valid for task_type "{TASK}". '
                      f'Valid options: {VALID_METRICS[TASK]}')

print(f'use case: {CONFIG["use_case"]["name"]}  |  algorithm: {ALGO}  |  task_type: {TASK}')
print(f'{len(FEATURES)} features | {len(CAT_COLS)} categorical | {len(NUM_COLS)} numeric')
print('resolved object names:')
print(json.dumps(RESOLVED, indent=2))

# ============================================================================
# Cell: c2_load_split_encode
# ============================================================================
if CONFIG["data"].get("external_source_secret"):
    # Example only: pulling raw data from a NON-Snowflake source. Swap the
    # connector below (psycopg2, pyodbc, ...) for whatever your source
    # system needs. Credential handling stays identical either way — the
    # secret value never appears in config.yaml or in this notebook.
    creds = get_secret_username_password(CONFIG["data"]["external_source_secret"])
    print(f'external_source_secret configured for user "{creds.username}" — '
          f'wire up your source-specific connector here, then load into `df`.')
    # df = your_connector.read_sql(..., user=creds.username, password=creds.password)

df = session.table(f'{FQ}.{CONFIG["data"]["source_view"]}').to_pandas()
df.columns = [c.upper() for c in df.columns]

CLASS_LABELS = None  # populated only when a classification target is non-numeric (e.g. species names)

if TASK == "regression":
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    assert not df[TARGET].isna().any(), f'non-numeric or missing values found in target column "{TARGET}"'
    print(f'rows={len(df):,} | target mean={df[TARGET].mean():.4f} | target std={df[TARGET].std():.4f}')
else:
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        # label-encode string targets (e.g. "setosa"/"versicolor"/"virginica") into class codes
        CLASS_LABELS = sorted(df[TARGET].dropna().unique().tolist())
        label_map = {label: i for i, label in enumerate(CLASS_LABELS)}
        df[TARGET] = df[TARGET].map(label_map).astype("int8")
        print(f'target label encoding ({TARGET}): {label_map}')
    else:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype("int8")

    if TASK == "binary":
        base_rate = df[TARGET].mean()
        print(f'rows={len(df):,} | base_rate={base_rate:.4%} | imbalance 1:{(1-base_rate)/max(base_rate,1e-9):.0f}')
    else:
        print(f'rows={len(df):,} | class distribution:')
        print(df[TARGET].value_counts().sort_index().to_string())

missing = [c for c in FEATURES if c not in df.columns]
assert not missing, f"columns listed in config.yaml but not found in source: {missing}"

tr, te, split_point, strategy_used = split_data(df, CONFIG, SEED)

X_tr, y_tr = encode_features(tr, CONFIG), tr[TARGET]
FEATURE_ORDER = list(X_tr.columns)
X_te, y_te = encode_features(te, CONFIG, FEATURE_ORDER), te[TARGET]

MEDIAN_COLS = CONFIG["data"].get("median_impute_columns", [])
X_tr, X_te, IMPUTE_VALUES = impute_missing(X_tr, X_te, MEDIAN_COLS)

in_mask = split_hpo(tr, CONFIG)
X_in, y_in, X_va, y_va = X_tr[in_mask], y_tr[in_mask], X_tr[~in_mask], y_tr[~in_mask]

print(f'split strategy={strategy_used} @ {split_point} -> '
      f'train={len(tr):,} | test={len(te):,}')
print(f'HPO: fit={len(X_in):,} valid={len(X_va):,} | encoded features={len(FEATURE_ORDER)}')
assert not X_tr.isna().any().any(), "NaNs remain in X_tr after imputation"

# ============================================================================
# Cell: c3_hpo
# ============================================================================
BEST_PARAMS, N_ROUNDS, HPO_RESULTS = run_hpo(CONFIG, ALGO, X_in, y_in, X_va, y_va, SEED)

metric_name = CONFIG["model"]["optimization_metric"]
if TASK == "binary":
    baseline_desc = f'baseline {metric_name} (majority-class guess) = {y_va.mean():.4f}'
elif TASK == "multiclass":
    baseline_desc = f'baseline accuracy (majority-class guess) = {y_va.value_counts(normalize=True).max():.4f}'
else:  # regression
    baseline_desc = f'target std (naive baseline) = {y_va.std():.4f}'
print(baseline_desc)
print(f'best val {metric_name} = {HPO_RESULTS.iloc[0]["val_score"]:.4f}')
print(json.dumps(BEST_PARAMS, indent=2, default=str))
HPO_RESULTS.head(5)

# ============================================================================
# Cell: c4_fit_eval_explain
# ============================================================================
model = fit_final_model(CONFIG, ALGO, BEST_PARAMS, N_ROUNDS, X_tr, y_tr, SEED)

if TASK == "regression":
    pred_te = model.predict(X_te)
    pred_tr = model.predict(X_tr)
elif TASK == "binary":
    proba_te = model.predict_proba(X_te)[:, 1]
    proba_tr = model.predict_proba(X_tr)[:, 1]
    pred_te  = model.predict(X_te)
    pred_tr  = model.predict(X_tr)
else:  # multiclass
    proba_te = model.predict_proba(X_te)
    proba_tr = model.predict_proba(X_tr)
    pred_te  = model.predict(X_te)
    pred_tr  = model.predict(X_tr)

if TASK in ("binary", "multiclass"):
    confidence_te = proba_te if TASK == "binary" else proba_te.max(axis=1)
    CUT_HI, CUT_MD = risk_cutoffs(confidence_te,
                                   CONFIG["scoring"]["risk_band_high_n"],
                                   CONFIG["scoring"]["risk_band_medium_n"])
    THRESHOLD = CUT_HI
    cap = CONFIG["scoring"]["daily_capacity"]
    cap_thr = float(np.sort(confidence_te)[::-1][min(cap, len(confidence_te)) - 1])
else:
    CUT_HI = CUT_MD = THRESHOLD = cap = cap_thr = None  # risk banding is a classification-only concept

yv = y_te.values

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report
)

if TASK == "regression":
    METRICS = {
        "rmse": mean_squared_error(yv, pred_te, squared=False),
        "mae": mean_absolute_error(yv, pred_te),
        "r2": r2_score(yv, pred_te),
    }
elif TASK == "binary":
    METRICS = {
        "roc_auc": roc_auc_score(yv, proba_te),
        "pr_auc": average_precision_score(yv, proba_te),
        "f1": f1_score(yv, pred_te),
    }
else:  # multiclass
    METRICS = {
        "f1": f1_score(yv, pred_te, average="macro"),
        "accuracy": accuracy_score(yv, pred_te),
    }
    print(classification_report(yv, pred_te))

print(json.dumps(METRICS, indent=2))

if TASK in ("binary", "multiclass"):
    print(f'risk band: HIGH >= {CUT_HI:.4f} | MEDIUM >= {CUT_MD:.4f} | '
          f'capacity threshold ({cap}/day) = {cap_thr:.4f}')
else:
    print('risk bands / capacity threshold not applicable for regression')

ENC2BASE = build_enc2base_map(FEATURE_ORDER, CAT_COLS)

if CONFIG["explainability"]["enabled"]:
    import matplotlib
    matplotlib.use("Agg")  # headless script: no display backend, so plots are built but never shown
    import matplotlib.pyplot as plt

    SHAP_TR = feature_contributions(ALGO, model, X_tr, FEATURE_ORDER, ENC2BASE)
    imp = SHAP_TR.abs().mean().sort_values(ascending=False)
    top_n = CONFIG["explainability"]["top_n_features"]
    TOP = imp.head(top_n).index[::-1]
    arah = SHAP_TR[TOP].mean()

    effect_label = "predicted value" if TASK == "regression" else "target probability"

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].barh(TOP, imp[TOP], color="#4C72B0")
    ax[0].set_title("Feature strength (mean |contribution|)")
    ax[0].set_xlabel(f"mean |contribution| to {effect_label}")

    warna = ["#C44E52" if v > 0 else "#55A868" for v in arah]
    ax[1].barh(TOP, arah, color=warna)
    ax[1].axvline(0, color="black", lw=0.8)
    ax[1].set_title(f"Direction of effect (red = increases {effect_label})")
    ax[1].set_xlabel(f"mean contribution to {effect_label}")
    plt.tight_layout(); plt.show()

    tiga = imp.head(3).index
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, f in enumerate(tiga):
        xcol = f if f in X_tr.columns else [c for c in FEATURE_ORDER if ENC2BASE[c] == f][0]
        ax[i].scatter(X_tr[xcol], SHAP_TR[f], s=4, alpha=0.25, c=SHAP_TR[f], cmap="coolwarm",
                      vmin=-SHAP_TR[f].abs().max(), vmax=SHAP_TR[f].abs().max())
        ax[i].axhline(0, color="black", lw=0.8)
        ax[i].set_xlabel(f); ax[i].set_ylabel("contribution")
        ax[i].set_title(f"Effect of {f}")
    plt.tight_layout(); plt.show()

    print(f"\ntop {top_n} drivers (positive = increases {effect_label}):")
    print(pd.DataFrame({"mean_abs_contribution": imp.head(top_n), "direction": arah}).to_string())
else:
    print("explainability.enabled is false - skipping SHAP analysis")

# ============================================================================
# Cell: c5_feature_store
# ============================================================================
if CONFIG.get("feature_store", {}).get("enabled", False):
    from snowflake.ml.feature_store import FeatureStore, Entity, FeatureView, CreationMode

    fs = FeatureStore(session=session, database=CONFIG["snowflake"]["database"],
                      name=CONFIG["snowflake"]["schema"], default_warehouse=CONFIG["snowflake"]["warehouse"],
                      creation_mode=CreationMode.CREATE_IF_NOT_EXIST)

    entity = Entity(name=RESOLVED["entity_name"], join_keys=CONFIG["output"]["entity_join_keys"],
                    desc=f'Entity for {CONFIG["use_case"]["name"]}')
    fs.register_entity(entity)

    fv_ts_col = CONFIG["feature_store"].get("timestamp_column", TS_COL)
    feature_df = session.table(f'{FQ}.{CONFIG["data"]["source_view"]}').select(
        *CONFIG["output"]["entity_join_keys"], fv_ts_col, *FEATURES)

    fv = FeatureView(name=RESOLVED["feature_view_name"], entities=[entity], feature_df=feature_df,
                     timestamp_col=fv_ts_col, refresh_freq=None,
                     desc=f'{len(FEATURES)} features for {CONFIG["use_case"]["name"]}')
    fv = fs.register_feature_view(feature_view=fv, version=RESOLVED["model_version"], overwrite=True)

    print("entities:", [r["NAME"] for r in fs.list_entities().collect()])
    print("feature views:", [(r["NAME"], r["VERSION"]) for r in fs.list_feature_views().collect()])
else:
    print("feature_store.enabled is false - skipping feature store registration")

# ============================================================================
# Cell: c6_registry_contract
# ============================================================================
from snowflake.ml.registry import Registry

reg = Registry(session, database_name=CONFIG["snowflake"]["database"], schema_name=CONFIG["snowflake"]["schema"])
try:
    existing = [v.version_name.upper() for v in reg.get_model(RESOLVED["model_name"]).versions()]
except Exception:
    existing = []

if RESOLVED["model_version"].upper() in existing:
    mv = reg.get_model(RESOLVED["model_name"]).version(RESOLVED["model_version"])
    print(f'version {RESOLVED["model_version"]} already exists - reusing as-is')
else:
    threshold_note = f'threshold={THRESHOLD:.4f}' if THRESHOLD is not None else 'threshold=n/a (regression)'
    mv = reg.log_model(
        model, model_name=RESOLVED["model_name"], version_name=RESOLVED["model_version"],
        sample_input_data=X_tr.head(100).astype("float64"),
        metrics=METRICS, options={"enable_explainability": True},
        target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
        comment=(f'{ALGO} | task_type={TASK} | {CONFIG["model"]["n_hpo_trials"]} HPO trials | '
                 f'{threshold_note}'),
    )
    print(f'version {RESOLVED["model_version"]} registered')

FNS = [f["name"].upper() for f in mv.show_functions()]
print("functions:", FNS)

log_feature_contract(session, CONFIG, RESOLVED, FEATURE_ORDER, ENC2BASE)
log_run(session, CONFIG, RESOLVED, METRICS, extra={
    "best_params": BEST_PARAMS, "n_estimators": N_ROUNDS, "impute_values": IMPUTE_VALUES,
    "risk_cut_high": CUT_HI, "risk_cut_medium": CUT_MD, "capacity_threshold": cap_thr,
})
print(f'{RESOLVED["contract_table"]} and {RESOLVED["config_table"]} updated automatically - no SQL written by hand.')

# ============================================================================
# Cell: c7_baseline_monitor
# ============================================================================
contrib_te = feature_contributions(ALGO, model, X_te, FEATURE_ORDER, ENC2BASE)
A, cols = contrib_te.values, np.array(contrib_te.columns)
n_causes = CONFIG["explainability"]["top_n_causes_per_row"]
idx = np.argsort(-A, axis=1)[:, :n_causes]
causes = [" | ".join(cols[r][A[i, r] > 0]) for i, r in enumerate(idx)]

horizon = CONFIG["scoring"]["label_horizon_days"]
scored_at = pd.to_datetime(te[TS_COL].astype(str) + "01", format="%Y%m%d", errors="coerce")
if scored_at.isna().all():
    scored_at = pd.Series(pd.Timestamp.now(), index=te.index)

base = pd.concat([te[KEYS].reset_index(drop=True), X_te.reset_index(drop=True)], axis=1)

if TASK == "regression":
    base["PREDICTED_VALUE"] = pred_te
    base["PREDICTION"]      = pred_te
    base["RISK_BAND"]       = None
elif TASK == "binary":
    base["PREDICTED_PROBABILITY"] = proba_te
    base["PREDICTION"]            = (proba_te >= THRESHOLD).astype("int8")
    base["RISK_BAND"]             = risk_band(proba_te, CUT_HI, CUT_MD)
else:  # multiclass
    base["PREDICTED_PROBABILITY"] = proba_te.max(axis=1)
    base["PREDICTION"]            = pred_te
    base["RISK_BAND"]             = risk_band(proba_te.max(axis=1), CUT_HI, CUT_MD)

base["PROBABLE_CAUSES"]       = causes
base["ACTUAL_LABEL"]          = yv
base["MODEL_VERSION"]         = RESOLVED["model_version"]
base["SCORED_AT"]       = scored_at.dt.strftime("%Y-%m-%d %H:%M:%S").values
base["LABEL_MATURE_AT"] = (scored_at + pd.Timedelta(days=horizon)).dt.strftime("%Y-%m-%d %H:%M:%S").values

session.sql(f'DROP TABLE IF EXISTS {FQ}.{RESOLVED["baseline_table"]}').collect()
session.write_pandas(base, RESOLVED["baseline_table"], database=CONFIG["snowflake"]["database"],
                     schema=CONFIG["snowflake"]["schema"], auto_create_table=True, overwrite=True,
                     quote_identifiers=False)

session.sql(f'''
CREATE OR REPLACE TABLE {FQ}.{RESOLVED["baseline_table"]} AS
SELECT * EXCLUDE (SCORED_AT, LABEL_MATURE_AT),
       SCORED_AT::TIMESTAMP_NTZ(6)       AS SCORED_AT,
       LABEL_MATURE_AT::TIMESTAMP_NTZ(6) AS LABEL_MATURE_AT
FROM {FQ}.{RESOLVED["baseline_table"]}''').collect()

session.sql(f'DROP TABLE IF EXISTS {FQ}.{RESOLVED["monitor_table"]}').collect()
session.sql(f'CREATE TABLE {FQ}.{RESOLVED["monitor_table"]} LIKE {FQ}.{RESOLVED["baseline_table"]}').collect()
session.sql(f'INSERT INTO {FQ}.{RESOLVED["monitor_table"]} SELECT * FROM {FQ}.{RESOLVED["baseline_table"]}').collect()

print(f'{RESOLVED["baseline_table"]}: {len(base):,} rows, {base.shape[1]} columns')

if TASK == "regression":
    session.sql(f'''
        SELECT SCORED_AT::DATE AS DT, COUNT(*) AS N,
               AVG(PREDICTION) AS AVG_PREDICTED_VALUE
        FROM {FQ}.{RESOLVED["monitor_table"]} GROUP BY 1 ORDER BY 1''').show()
else:
    session.sql(f'''
        SELECT SCORED_AT::DATE AS DT, COUNT(*) AS N, SUM(IFF(RISK_BAND = 'HIGH', 1, 0)) AS N_HIGH
        FROM {FQ}.{RESOLVED["monitor_table"]} GROUP BY 1 ORDER BY 1''').show()

# ============================================================================
# Cell: c8_model_monitor
# ============================================================================
if CONFIG.get("model_monitor", {}).get("enabled", False):
    monitor_name = f'{RESOLVED["model_name"]}_MONITOR_{RESOLVED["model_version"]}'
    create_or_replace_model_monitor(
        session, CONFIG, monitor_name, RESOLVED["model_name"], RESOLVED["model_version"],
        RESOLVED["baseline_table"], RESOLVED["monitor_table"], KEYS[0],
    )
    print(f'model monitor {monitor_name} created/updated, watching {RESOLVED["model_version"]}')
else:
    print("model_monitor.enabled is false - skipping model monitor creation")
