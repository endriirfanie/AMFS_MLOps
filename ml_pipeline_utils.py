"""
ml_pipeline_utils.py

Reusable helpers for the generic ML pipeline notebook. Everything that is
Snowflake-specific, config-parsing, or algorithm-specific lives here so the
notebook itself stays short, readable, and identical across use cases.

Put this file in the same Workspace directory as the notebook and
config.yaml — Snowflake Notebooks in Workspaces import local .py files
directly (`from ml_pipeline_utils import *`).

Supports three task types (config.yaml: model.task_type): "binary",
"multiclass", "regression". Every function that behaves differently per
task type takes task_type explicitly rather than re-deriving it, so the
notebook (which reads it once from CONFIG) is the single source of truth.
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

SUPPORTED_ALGORITHMS = ("xgboost", "lightgbm", "random_forest", "logistic_regression")
VALID_TASK_TYPES = ("binary", "multiclass", "regression")
VALID_OPTIMIZATION_METRICS = {
    "binary": {"pr_auc", "roc_auc", "f1"},
    "multiclass": {"f1"},
    "regression": {"rmse", "mae", "r2"},
}


# ============================================================================
# Config
# ============================================================================
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg)
    # Not part of the schema — lets helpers (run_sql_script, etc.) resolve
    # use-case-owned files relative to config.yaml's own folder, regardless
    # of where the notebook itself lives (training/retraining notebooks can
    # sit in a shared ML Setup folder while config.yaml sits per use case).
    # Stripped out again before anything is written to Snowflake.
    cfg["_meta"] = {
        "config_path": os.path.abspath(path),
        "config_dir": os.path.dirname(os.path.abspath(path)),
    }
    return cfg


def _strip_meta(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k != "_meta"}


def _validate_config(cfg: dict) -> None:
    required = ["use_case", "snowflake", "data", "split", "model", "scoring", "output"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config.yaml is missing required section(s): {missing}")
    algo = cfg["model"]["algorithm"]
    if algo not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"model.algorithm '{algo}' not supported. Choose one of {SUPPORTED_ALGORITHMS}")
    if algo not in cfg["model"].get("search_space", {}):
        raise ValueError(f"model.search_space has no entry for algorithm '{algo}'")

    # use cases created before task_type existed are all binary: default it rather than break them
    task_type = cfg["model"].setdefault("task_type", "binary")
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(f"model.task_type '{task_type}' is invalid. Valid options: {VALID_TASK_TYPES}")
    opt_metric = cfg["model"].get("optimization_metric")
    if opt_metric not in VALID_OPTIMIZATION_METRICS[task_type]:
        raise ValueError(f"optimization_metric '{opt_metric}' is not valid for task_type '{task_type}'. "
                          f"Valid options: {VALID_OPTIMIZATION_METRICS[task_type]}")

    retrain = cfg.get("retraining")
    if retrain and retrain.get("enabled"):
        for k in ("previous_version", "new_version", "promotion"):
            if k not in retrain:
                raise ValueError(f"retraining.{k} is required when retraining.enabled is true")


def resolve_names(cfg: dict) -> dict:
    """Fill in any output.* names left null, derived from use_case.prefix/name."""
    prefix = cfg["use_case"]["prefix"].upper()
    name = cfg["use_case"]["name"].upper()
    out = cfg["output"]
    inf = cfg.get("inference", {}) or {}
    defaults = {
        "model_name": f"{prefix}_{name}_MODEL",
        "feature_store_name": f"{prefix}_FEATURE_STORE",
        "entity_name": f"{prefix}_ENTITY",
        "feature_view_name": f"{prefix}_FEATURES",
        "baseline_table": f"{prefix}_MONITOR_BASELINE",
        "monitor_table": f"{prefix}_PREDICT_MONITOR",
    }
    resolved = {k: (out.get(k) or v) for k, v in defaults.items()}
    resolved["model_version"] = out["model_version"]
    resolved["contract_table"] = out["contract_table"]
    resolved["config_table"] = out["config_table"]
    # Shared across all use cases, like contract_table/config_table.
    resolved["log_table"] = out.get("log_table") or "ML_SCORING_LOG"
    resolved["watermark_table"] = out.get("watermark_table") or "ML_SCORING_WATERMARK"
    # Only meaningful to generic_ml_inference_deploy_pipeline.ipynb.
    resolved["input_view"] = inf.get("input_view") or f"{prefix}_SCORE_INPUT"
    resolved["procedure_name"] = inf.get("procedure_name") or f"{prefix}_SP_SCORE_DAILY"
    # Only meaningful to generic_ml_ground_truth_deploy_pipeline.ipynb.
    resolved["gt_procedure_name"] = (cfg.get("retraining") or {}).get("ground_truth_procedure_name") or f"{prefix}_SP_GROUND_TRUTH"
    return resolved


# ============================================================================
# Session / secrets
# ============================================================================
def get_session(cfg: dict):
    """Reuse the active Snowflake session when running inside a Snowflake
    Notebook / Workspace (the normal case). Falls back to an explicit
    connector session for local/dev execution, only when
    snowflake.external_connection.enabled is true in config.yaml.
    """
    from snowflake.snowpark.context import get_active_session

    try:
        return get_active_session()
    except Exception:
        pass

    ext = (cfg["snowflake"].get("external_connection") or {})
    if not ext.get("enabled"):
        raise RuntimeError(
            "No active Snowflake session found. Run this notebook inside Snowflake "
            "Notebooks in Workspaces, or set snowflake.external_connection.enabled: true "
            "in config.yaml to connect from outside Snowflake."
        )

    from snowflake.snowpark import Session

    conn_params = {
        "account": ext.get("account") or os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "role": cfg["snowflake"].get("role") or os.environ.get("SNOWFLAKE_ROLE"),
        "warehouse": cfg["snowflake"]["warehouse"],
        "database": cfg["snowflake"]["database"],
        "schema": cfg["snowflake"]["schema"],
    }
    # Prefer key-pair auth; password is a fallback for local/dev only.
    # Neither the key path nor the password should ever be written into
    # config.yaml — they come from the environment (or a secrets manager
    # your orchestrator injects), never from a file checked into git.
    if os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH"):
        conn_params["private_key_file"] = os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"]
    else:
        conn_params["password"] = os.environ["SNOWFLAKE_PASSWORD"]
    return Session.builder.configs(conn_params).create()


def get_secret_string(secret_name: str) -> str:
    """Fetch a GENERIC_STRING secret. Only works when running inside a
    Snowflake-hosted notebook/service that has the secret attached."""
    from snowflake.snowpark.secrets import get_generic_secret_string
    return get_generic_secret_string(secret_name)


def get_secret_username_password(secret_name: str):
    """Fetch a PASSWORD-type secret. Returns an object with .username/.password.
    Only works when running inside a Snowflake-hosted notebook/service that
    has the secret attached (see setup_secrets.sql + README.md)."""
    from snowflake.snowpark.secrets import get_username_password
    return get_username_password(secret_name)


# ============================================================================
# Feature encoding
# ============================================================================
def normalize_categorical(s: pd.Series) -> pd.Series:
    return (
        s.astype("string").str.strip().str.upper()
        .str.replace(r"[^A-Z0-9]+", "_", regex=True)
        .str.strip("_").replace({"": pd.NA}).fillna("MISSING")
    )


def encode_features(df: pd.DataFrame, cfg: dict, columns=None) -> pd.DataFrame:
    cat_cols = cfg["data"]["categorical_columns"]
    num_cols = cfg["data"]["numeric_columns"]
    features = cat_cols + num_cols
    X = df[features].copy()
    for c in cat_cols:
        X[c] = normalize_categorical(X[c])
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = pd.get_dummies(X, columns=cat_cols, dtype="float64")
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X.astype("float64")


def build_enc2base_map(feature_order, cat_cols):
    enc2base = {}
    for e in feature_order:
        base = next((c for c in sorted(cat_cols, key=len, reverse=True)
                     if e == c or e.startswith(c + "_")), e)
        enc2base[e] = base
    return enc2base


def impute_missing(X_tr: pd.DataFrame, X_te: pd.DataFrame, median_cols=None):
    median_cols = set(median_cols or [])
    impute_values = {}
    for c in X_tr.columns:
        if X_tr[c].isna().any():
            v = X_tr[c].median() if c in median_cols else X_tr[c].mean()
            impute_values[c] = float(0.0 if pd.isna(v) else v)
    X_tr = X_tr.fillna(impute_values).astype("float64")
    X_te = X_te.fillna(impute_values).astype("float64")
    return X_tr, X_te, impute_values


# ============================================================================
# Use-case-owned SQL hooks (retraining: ground truth, retrain-base view)
# ============================================================================
def run_sql_script(session, path: str, params: dict = None):
    """Executes a plain .sql file, one statement per top-level ';'. These
    files are use-case-owned (ground-truth labeling, retrain-base views) —
    keep statements simple; this splitter does not understand semicolons
    inside string literals or comments. `params` does simple {name} substitution."""
    with open(path, "r") as f:
        script = f.read()
    if params:
        for k, v in params.items():
            script = script.replace(f"{{{k}}}", str(v))
    results = []
    for raw in script.split(";"):
        lines = raw.strip().splitlines()
        # drop leading blank / comment lines so a statement that follows a comment still runs
        while lines and (not lines[0].strip() or lines[0].strip().startswith("--")):
            lines.pop(0)
        stmt = "\n".join(lines).strip()
        if not stmt:
            continue
        results.append(session.sql(stmt).collect())
    return results


# ============================================================================
# Retraining: ground truth, retrain dataset, champion comparison helpers
#
# Everything in this section is generic. It is driven by config.yaml's
# `retraining:` section and by the fixed monitor-table column contract, so a
# new use case only has to say WHERE its true outcomes come from
# (retraining.ground_truth_source, a table/view holding the key and the
# outcome). It does not write merge or union SQL of its own.
# ============================================================================
def sql_params(cfg: dict, resolved: dict) -> dict:
    """{name} placeholders available to use-case-owned .sql files, so those
    files never hardcode a database, schema or table name (see run_sql_script)."""
    rt, inf = cfg.get("retraining") or {}, cfg.get("inference") or {}
    db, schema = cfg["snowflake"]["database"], cfg["snowflake"]["schema"]
    return {
        "DB": db, "SCHEMA": schema, "FQ": f"{db}.{schema}",
        "MONITOR_TABLE": resolved["monitor_table"],
        "TRAIN_SOURCE": cfg["data"]["source_view"],
        "PROD_FEED": rt.get("production_source") or inf.get("source_stream_view") or "",
        "WATERMARK_COL": inf.get("watermark_column") or "",
        "GT_SOURCE": rt.get("ground_truth_source") or "",
    }


def next_version_name(prev: str) -> str:
    """XGB_R1 -> XGB_R2, XGB_R1_GENERIC -> XGB_R2_GENERIC. A name without an
    R<number> part just gets _R2 appended."""
    import re
    hits = list(re.finditer(r"(?:(?<=_)|^)R(\d+)(?=_|$)", prev, flags=re.I))
    if not hits:
        return f"{prev}_R2"
    h = hits[-1]
    return prev[:h.start()] + f"R{int(h.group(1)) + 1}" + prev[h.end():]


def registry_versions(session, cfg: dict, resolved: dict) -> list:
    """Upper-cased version names already registered under this use case's model."""
    from snowflake.ml.registry import Registry
    reg = Registry(session, database_name=cfg["snowflake"]["database"],
                   schema_name=cfg["snowflake"]["schema"])
    try:
        return [v.version_name.upper() for v in reg.get_model(resolved["model_name"]).versions()]
    except Exception:
        return []


def resolve_retrain_versions(session, cfg: dict, resolved: dict):
    """(champion version, challenger version). A null previous_version means
    the latest run logged for this use case; a null new_version means the
    champion's R-number plus one."""
    rt = cfg["retraining"]
    prev = rt.get("previous_version")
    if not prev:
        last = load_past_run(session, cfg, resolved)
        if last is None:
            raise ValueError(f'no earlier run found in {resolved["config_table"]} for '
                             f'{cfg["use_case"]["name"]} - run the training notebook first.')
        prev = last["model_version"]
    new = rt.get("new_version") or next_version_name(prev)
    if new.upper() == prev.upper():
        raise ValueError(f"retraining.new_version must differ from the champion version ({prev}).")
    return prev, new


def prepare_target(df: pd.DataFrame, target: str, task_type: str):
    """Returns (df, class_labels). Regression keeps a numeric target and refuses
    non-numeric values. Classification gives int8 class codes: a numeric (or
    all-digit text) target is used as is, anything else is label-encoded in sorted
    order, and the label list is returned so the mapping can be reported."""
    df = df.copy()
    if task_type == "regression":
        df[target] = pd.to_numeric(df[target], errors="coerce")
        if df[target].isna().any():
            raise ValueError(f'non-numeric or missing values in regression target "{target}"')
        return df, None
    as_num = pd.to_numeric(df[target], errors="coerce")
    if pd.api.types.is_numeric_dtype(df[target]) or as_num.notna().sum() == df[target].notna().sum():
        df[target] = as_num.fillna(0).astype("int8")
        return df, None
    labels = sorted(df[target].dropna().unique().tolist())
    df[target] = df[target].map({lab: i for i, lab in enumerate(labels)}).fillna(0).astype("int8")
    return df, labels


def load_champion_predictions(session, cfg: dict, resolved: dict, key_col: str,
                              model_version: str = None) -> pd.DataFrame:
    """Previously scored rows that now carry a real outcome, from the live
    monitor table: one row per key (the latest score wins), optionally limited
    to one model version. Relies on the fixed column contract every generic
    notebook writes. Columns: key, the score column (PREDICTED_PROBABILITY, or
    PREDICTED_VALUE for regression), PREDICTION, ACTUAL_LABEL, SCORED_AT."""
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}.{resolved["monitor_table"]}'
    score_col = ("PREDICTED_VALUE" if cfg["model"].get("task_type", "binary") == "regression"
                 else "PREDICTED_PROBABILITY")
    t = session.table(fq)
    t = t.filter(t["ACTUAL_LABEL"].is_not_null())
    if model_version:
        t = t.filter(t["MODEL_VERSION"] == model_version)
    df = t.select(key_col, score_col, "PREDICTION", "ACTUAL_LABEL", "SCORED_AT").to_pandas()
    df.columns = [c.upper() for c in df.columns]
    return (df.sort_values("SCORED_AT", kind="mergesort")
              .drop_duplicates(subset=[key_col.upper()], keep="last").reset_index(drop=True))


def _log_event(session, cfg: dict, resolved: dict, procedure: str, n_rows: int, status: str, message: str):
    """One row in the shared scoring log (same 6 positional columns the scoring
    procedure writes). Logging never fails the caller."""
    fq_log = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}.{resolved["log_table"]}'
    q = lambda s: str(s).replace("'", "''")
    try:
        session.sql(
            f"INSERT INTO {fq_log} VALUES (CURRENT_TIMESTAMP()::TIMESTAMP_NTZ(6), "
            f"'{q(cfg['use_case']['name'])}', '{q(procedure)}', {int(n_rows)}, '{q(status)}', "
            f"'{q(message)[:2000]}')").collect()
    except Exception as e:
        print(f"(could not write to {resolved['log_table']}: {e})")


def build_ground_truth_merge_sql(cfg: dict, resolved: dict) -> str:
    """MERGE that back-fills ACTUAL_LABEL in the monitor table from the
    use-case's ground-truth source (retraining.ground_truth_source: a table or
    view with one row per key holding the true outcome).

    Safe to repeat: only rows whose ACTUAL_LABEL is still NULL are touched.
    With retraining.respect_label_maturity (default true) a row is only labeled
    once its LABEL_MATURE_AT has passed. A deliberately narrow contract so a use
    case never writes merge SQL, only the source that says what really happened."""
    rt = cfg["retraining"]
    task_type = cfg["model"].get("task_type", "binary")
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}'
    src, label = rt["ground_truth_source"], rt["ground_truth_label_column"]
    key = rt.get("ground_truth_key_column") or cfg["data"]["key_columns"][0]
    ts = rt.get("ground_truth_ts_column")
    label_expr = {"binary": f"{label}::NUMBER(38,0)", "multiclass": f"{label}::VARCHAR",
                  "regression": f"{label}::FLOAT"}[task_type]
    dedupe = f"\n    QUALIFY ROW_NUMBER() OVER (PARTITION BY {key} ORDER BY {ts} DESC) = 1" if ts else ""
    matured = " AND t.LABEL_MATURE_AT <= CURRENT_TIMESTAMP()" if rt.get("respect_label_maturity", True) else ""
    return (f"MERGE INTO {fq}.{resolved['monitor_table']} t\n"
            f"USING (\n"
            f"    SELECT {key} AS GT_KEY, {label_expr} AS GT_LABEL\n"
            f"    FROM {fq}.{src}\n"
            f"    WHERE {label} IS NOT NULL{dedupe}\n"
            f") s ON t.{key} = s.GT_KEY\n"
            f"WHEN MATCHED AND t.ACTUAL_LABEL IS NULL{matured}\n"
            f"    THEN UPDATE SET t.ACTUAL_LABEL = s.GT_LABEL")


def merge_ground_truth(session, cfg: dict, resolved: dict) -> dict:
    """Runs build_ground_truth_merge_sql and reports what changed."""
    rt = cfg["retraining"]
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}'
    mon, src = f"{fq}.{resolved['monitor_table']}", f"{fq}.{rt['ground_truth_source']}"
    key = rt.get("ground_truth_key_column") or cfg["data"]["key_columns"][0]
    label = rt["ground_truth_label_column"]

    if not rt.get("ground_truth_ts_column"):
        dup = session.sql(f"SELECT COUNT(*) - COUNT(DISTINCT {key}) AS D FROM {src} "
                          f"WHERE {label} IS NOT NULL").to_pandas()["D"].iloc[0]
        if int(dup) > 0:
            raise ValueError(f"{rt['ground_truth_source']} has {int(dup)} duplicate {key} value(s), so the "
                             f"merge would be ambiguous. Make it unique, or set "
                             f"retraining.ground_truth_ts_column so the latest row wins.")

    def counts():
        r = session.sql(f"SELECT COUNT(*) AS N, COUNT_IF(ACTUAL_LABEL IS NOT NULL) AS L FROM {mon}").to_pandas()
        return int(r["N"].iloc[0]), int(r["L"].iloc[0])

    total, before = counts()
    session.sql(build_ground_truth_merge_sql(cfg, resolved)).collect()
    _, after = counts()
    out = {"total_rows": total, "labeled_before": before, "labeled_after": after,
           "newly_labeled": after - before, "still_unlabeled": total - after}
    _log_event(session, cfg, resolved, "GROUND_TRUTH_MERGE", out["newly_labeled"], "OK",
               f"labeled {out['newly_labeled']} rows; {out['still_unlabeled']} still waiting")
    return out


def check_ground_truth_source(session, cfg: dict, resolved: dict) -> dict:
    """Validates the use case's ground-truth source before anything depends on it:
    it must exist, expose the key and outcome columns, and (unless a timestamp
    column lets the latest row win) hold one row per key. Also reports how many
    monitor rows are still waiting and how many of those the source can label."""
    rt = cfg["retraining"]
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}'
    src = rt.get("ground_truth_source")
    if not src:
        raise ValueError("retraining.ground_truth_source is not set in config.yaml")
    key = rt.get("ground_truth_key_column") or cfg["data"]["key_columns"][0]
    label, ts = rt["ground_truth_label_column"], rt.get("ground_truth_ts_column")

    cols = list(session.table(f"{fq}.{src}").columns)
    plain = {c.upper() for c in cols if not c.startswith('"')}
    missing = [c for c in [key, label] + ([ts] if ts else []) if c.upper() not in plain]
    if missing:
        raise ValueError(f"{src} is missing column(s) {missing}. Columns it has: {cols[:30]}")

    s = session.sql(f"SELECT COUNT(*) AS N, COUNT(DISTINCT {key}) AS K, COUNT_IF({label} IS NULL) AS NULLS "
                    f"FROM {fq}.{src}").to_pandas()
    rows, keys, null_labels = int(s["N"].iloc[0]), int(s["K"].iloc[0]), int(s["NULLS"].iloc[0])
    if not ts and rows - keys > 0:
        raise ValueError(f"{src} has {rows - keys} duplicate {key} value(s), so the merge would be ambiguous. "
                         f"Make it unique, or set retraining.ground_truth_ts_column so the latest row wins.")
    m = session.sql(
        f"SELECT COUNT(*) AS N_MON, COUNT_IF(m.ACTUAL_LABEL IS NULL) AS N_PENDING, "
        f"COUNT_IF(m.ACTUAL_LABEL IS NULL AND g.GT_KEY IS NOT NULL) AS N_LABELABLE "
        f"FROM {fq}.{resolved['monitor_table']} m "
        f"LEFT JOIN (SELECT DISTINCT {key} AS GT_KEY FROM {fq}.{src} WHERE {label} IS NOT NULL) g "
        f"ON m.{key} = g.GT_KEY").to_pandas()
    return {"source_rows": rows, "distinct_keys": keys, "null_outcomes": null_labels,
            "monitor_rows": int(m["N_MON"].iloc[0]), "monitor_pending": int(m["N_PENDING"].iloc[0]),
            "labelable_now": int(m["N_LABELABLE"].iloc[0])}


def build_ground_truth_procedure_sql(cfg: dict, resolved: dict) -> str:
    """CREATE PROCEDURE for the scheduled ground-truth back-fill, so an external
    orchestrator (Airflow) can run it with a plain zero-argument CALL, the same
    way it triggers scoring. It runs the generic merge (build_ground_truth_merge_sql),
    which only touches rows whose ACTUAL_LABEL is still NULL, so repeating it is safe.

    Returns 'OK: labeled N rows, M still waiting'. Zero new rows is a normal
    success (no outcomes arrived yet). Any failure is written to the scoring log
    and re-raised, so the orchestrator sees an ordinary task failure.

    Created EXECUTE AS OWNER: the caller only needs USAGE on the procedure. The
    owner role needs UPDATE on the monitor table and SELECT on the source.
    The merge settings are baked into the DDL, so re-deploy after changing them."""
    rt = cfg["retraining"]
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}'
    proc, uc = resolved["gt_procedure_name"], cfg["use_case"]["name"]
    mon, log = f"{fq}.{resolved['monitor_table']}", f"{fq}.{resolved['log_table']}"
    src = f"{fq}.{rt['ground_truth_source']}"
    key = rt.get("ground_truth_key_column") or cfg["data"]["key_columns"][0]
    label = rt["ground_truth_label_column"]
    merge_sql = build_ground_truth_merge_sql(cfg, resolved)
    dup_check = "" if rt.get("ground_truth_ts_column") else (
        f"    SELECT COUNT(*) - COUNT(DISTINCT {key}) INTO n_dup FROM {src} WHERE {label} IS NOT NULL;\n"
        f"    IF (n_dup > 0) THEN\n"
        f"        RAISE dup_keys;\n"
        f"    END IF;\n")
    return f"""
CREATE OR REPLACE PROCEDURE {fq}.{proc}()
RETURNS STRING LANGUAGE SQL
EXECUTE AS OWNER
AS
$$
DECLARE
    run_ts   TIMESTAMP_NTZ(6);
    n_total  NUMBER DEFAULT 0;
    n_before NUMBER DEFAULT 0;
    n_after  NUMBER DEFAULT 0;
    n_dup    NUMBER DEFAULT 0;
    dup_keys EXCEPTION (-20101, 'ground truth source has duplicate keys, so the merge would be ambiguous');
BEGIN
    run_ts := CURRENT_TIMESTAMP()::TIMESTAMP_NTZ(6);
{dup_check}
    SELECT COUNT(*), COUNT_IF(ACTUAL_LABEL IS NOT NULL) INTO n_total, n_before FROM {mon};

    {merge_sql};

    SELECT COUNT_IF(ACTUAL_LABEL IS NOT NULL) INTO n_after FROM {mon};

    INSERT INTO {log}
    VALUES (:run_ts, '{uc}', '{proc}', :n_after - :n_before, 'OK',
            'labeled ' || (:n_after - :n_before) || ' rows, ' || (:n_total - :n_after) || ' still waiting');

    RETURN 'OK: labeled ' || (:n_after - :n_before) || ' rows, ' || (:n_total - :n_after) || ' still waiting';

EXCEPTION
    WHEN OTHER THEN
        BEGIN
            INSERT INTO {log}
            VALUES (:run_ts, '{uc}', '{proc}', 0, 'FAILED', :sqlerrm);
        EXCEPTION
            WHEN OTHER THEN NULL;
        END;
        RAISE;
END;
$$
"""


def deploy_ground_truth_procedure(session, cfg: dict, resolved: dict) -> str:
    """Builds and executes the ground-truth procedure DDL. Returns the DDL for inspection."""
    ddl = build_ground_truth_procedure_sql(cfg, resolved)
    session.sql(ddl).collect()
    return ddl


def build_retrain_base_view_sql(cfg: dict, resolved: dict) -> str:
    """The dataset a challenger trains on, as one view: the original training
    data (ROW_SOURCE = 'HISTORY') plus production rows that now have a real
    outcome (ROW_SOURCE = 'PRODUCTION').

    A production row is the latest feed record for a key at or before the moment
    it was scored, joined to that key's latest labeled score in the monitor
    table. The label therefore lives in exactly one place (the monitor table,
    maintained by the ground-truth merge) and accumulates run after run instead
    of being rebuilt. SCORED_AT is exposed so the notebook can hold out the
    newest scored rows for the champion comparison. Needs only:
        retraining.production_timestamp_sql  the value data.timestamp_column takes for production rows,
                                             written with {m} = monitor row and {c} = feed row"""
    rt, d, inf = cfg["retraining"], cfg["data"], cfg.get("inference") or {}
    task_type = cfg["model"].get("task_type", "binary")
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}'
    view = rt["retrain_base_view"]
    feed = rt.get("production_source") or inf.get("source_stream_view")
    if not feed:
        raise ValueError("no production feed: set retraining.production_source or inference.source_stream_view")
    ts_sql = rt.get("production_timestamp_sql")
    if not ts_sql:
        raise ValueError("retraining.production_timestamp_sql is required, for example "
                         "\"TO_NUMBER(TO_CHAR({m}.SCORED_AT, 'YYYYMM'))\" when data.timestamp_column is a YYYYMM cohort")
    keys, ts_col, target = list(d["key_columns"]), d["timestamp_column"], d["target_column"]
    feats = list(d["categorical_columns"]) + list(d["numeric_columns"])
    cols = list(dict.fromkeys(keys + [ts_col, target] + feats))
    label_expr = {"binary": "m.ACTUAL_LABEL::NUMBER(38,0)", "multiclass": "m.ACTUAL_LABEL::VARCHAR",
                  "regression": "m.ACTUAL_LABEL::FLOAT"}[task_type]

    hist = [f"{c}::VARCHAR AS {c}" if (c == target and task_type == "multiclass") else c for c in cols]

    def prod(c):
        if c == ts_col:
            return f"{ts_sql.replace('{m}', 'm').replace('{c}', 'c')} AS {c}"
        return f"{label_expr} AS {c}" if c == target else f"c.{c}"

    join_on = " AND ".join(f"m.{k} = c.{k}" for k in keys)
    wm = inf.get("watermark_column")
    latest_feed = ""
    if wm:
        latest_feed = (f"\n    WHERE c.{wm} <= m.SCORED_AT\n"
                       f"    QUALIFY ROW_NUMBER() OVER (PARTITION BY {', '.join('c.' + k for k in keys)} "
                       f"ORDER BY c.{wm} DESC) = 1")
    return (f"CREATE OR REPLACE VIEW {fq}.{view} AS\n"
            f"SELECT {', '.join(hist)}, 'HISTORY' AS ROW_SOURCE, NULL::TIMESTAMP_NTZ(6) AS SCORED_AT\n"
            f"FROM {fq}.{d['source_view']}\n"
            f"UNION ALL\n"
            f"SELECT * FROM (\n"
            f"    SELECT {', '.join(prod(c) for c in cols)}, 'PRODUCTION' AS ROW_SOURCE, m.SCORED_AT AS SCORED_AT\n"
            f"    FROM {fq}.{feed} c\n"
            f"    JOIN (\n"
            f"        SELECT {', '.join(keys)}, ACTUAL_LABEL, SCORED_AT\n"
            f"        FROM {fq}.{resolved['monitor_table']}\n"
            f"        WHERE ACTUAL_LABEL IS NOT NULL\n"
            f"        QUALIFY ROW_NUMBER() OVER (PARTITION BY {', '.join(keys)} ORDER BY SCORED_AT DESC) = 1\n"
            f"    ) m ON {join_on}{latest_feed}\n"
            f")")


def create_retrain_base_view(session, cfg: dict, resolved: dict) -> dict:
    """Creates (or refreshes) the retrain dataset view and returns its row
    count per ROW_SOURCE. A view is a definition, so this is safe to repeat."""
    session.sql(build_retrain_base_view_sql(cfg, resolved)).collect()
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}'
    df = session.sql(f'SELECT ROW_SOURCE, COUNT(*) AS N FROM {fq}.{cfg["retraining"]["retrain_base_view"]} '
                     f'GROUP BY 1').to_pandas()
    return {str(r["ROW_SOURCE"]): int(r["N"]) for _, r in df.iterrows()}


def pick_comparison_holdout(comp: pd.DataFrame, time_col: str, fraction: float, seed: int) -> pd.DataFrame:
    """The newest `fraction` of `comp` by scoring time. Rows scored in the same
    batch share a timestamp, so ties are broken with a seeded shuffle to keep
    the split reproducible. The remainder is left for the challenger to train on."""
    rng = np.random.default_rng(seed)
    c = comp.assign(_TIE=rng.random(len(comp))).sort_values([time_col, "_TIE"], kind="mergesort")
    n_hold = min(max(int(round(len(c) * fraction)), 0), len(c))
    return c.iloc[len(c) - n_hold:].drop(columns="_TIE")


def log_retrain_outcome(session, cfg: dict, resolved: dict, old_version: str, new_version: str,
                        status: str, message: str):
    """PROMOTED / HELD / SKIPPED, as a row an orchestrator can read to decide what to do next."""
    _log_event(session, cfg, resolved, "RETRAIN", 0, status, f"{old_version} -> {new_version}: {message}")


# ============================================================================
# Split
# ============================================================================
def split_data(df: pd.DataFrame, cfg: dict, seed: int):
    """Returns (train_df, test_df, split_point, strategy_used)."""
    ts_col = cfg["data"]["timestamp_column"]
    test_frac = cfg["split"]["test_fraction"]
    strategy = cfg["split"]["strategy"]
    split_point = None

    if strategy == "temporal":
        coh = np.sort(df[ts_col].unique())
        cum = df[ts_col].value_counts().reindex(coh).cumsum() / len(df)
        split_point = coh[np.searchsorted(cum.values, 1 - test_frac)]
        tr, te = df[df[ts_col] < split_point], df[df[ts_col] >= split_point]
        if len(te) == 0 or len(tr) == 0:
            strategy = "random"

    if strategy == "random":
        rng = np.random.default_rng(seed)
        id_col = cfg["data"]["key_columns"][-1]
        ids = df[id_col].unique()
        te_ids = set(rng.choice(ids, size=max(int(len(ids) * test_frac), 1), replace=False))
        m = df[id_col].isin(te_ids)
        tr, te = df[~m], df[m]

    return tr, te, split_point, strategy


def split_hpo(tr: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Boolean mask over `tr` splitting it into an HPO fit/validation set."""
    ts_col = cfg["data"]["timestamp_column"]
    frac = cfg["split"]["hpo_validation_fraction"]
    in_coh = np.sort(tr[ts_col].unique())
    if len(in_coh) > 1:
        cutoff = in_coh[max(int(len(in_coh) * (1 - frac)), 1)]
        return (tr[ts_col] < cutoff).values
    return np.arange(len(tr)) < int(len(tr) * (1 - frac))


# ============================================================================
# Model factory / fitting
# ============================================================================
def build_model(algorithm: str, params: dict, fixed_params: dict, seed: int, task_type: str = "binary"):
    fixed_params = dict(fixed_params)  # local copy — never mutate the caller's config-derived dict
    if algorithm == "xgboost":
        import xgboost as xgb
        default_eval_metric = {"binary": "aucpr", "multiclass": "mlogloss", "regression": "rmse"}[task_type]
        eval_metric = fixed_params.pop("eval_metric", default_eval_metric)
        cls = xgb.XGBRegressor if task_type == "regression" else xgb.XGBClassifier
        return cls(random_state=seed, n_jobs=-1, eval_metric=eval_metric, **fixed_params, **params)
    if algorithm == "lightgbm":
        import lightgbm as lgb
        cls = lgb.LGBMRegressor if task_type == "regression" else lgb.LGBMClassifier
        return cls(random_state=seed, n_jobs=-1, verbosity=-1, **fixed_params, **params)
    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestRegressor if task_type == "regression" else RandomForestClassifier
        return cls(random_state=seed, n_jobs=-1, **fixed_params, **params)
    if algorithm == "logistic_regression":
        if task_type == "regression":
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**fixed_params, **params)
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=seed, **fixed_params, **params)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def fit_model(algorithm, model, X_tr, y_tr, X_val=None, y_val=None, early_stopping_rounds=None):
    """Fits with early stopping + eval_set when the algorithm/validation set support it."""
    use_early_stop = algorithm in ("xgboost", "lightgbm") and X_val is not None
    if algorithm == "xgboost" and use_early_stop:
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    elif algorithm == "lightgbm" and use_early_stop:
        import lightgbm as lgb
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
    else:
        model.fit(X_tr, y_tr)
    return model


def best_iteration(algorithm, model):
    if algorithm == "xgboost":
        return int(getattr(model, "best_iteration", 0) or 0)
    if algorithm == "lightgbm":
        return int(getattr(model, "best_iteration_", 0) or 0)
    return None


def sample_params(space: dict, rng: np.random.Generator, spw: float = 1.0, task_type: str = "binary") -> dict:
    p = {}
    for k, values in space.items():
        if k == "use_scale_pos_weight":
            continue
        p[k] = values[int(rng.integers(len(values)))]
    # scale_pos_weight is a binary-imbalance concept only
    if task_type == "binary" and space.get("use_scale_pos_weight"):
        p["scale_pos_weight"] = [1.0, float(np.sqrt(spw)), spw][int(rng.integers(3))]
    return p


def run_hpo(cfg: dict, algorithm: str, X_in, y_in, X_val, y_val, seed: int):
    """Random-search HPO. Returns (best_params, n_rounds_for_final_fit, results_df)."""
    from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                                  mean_squared_error, mean_absolute_error, r2_score)

    task_type = cfg["model"].get("task_type", "binary")
    space = cfg["model"]["search_space"][algorithm]
    fixed = dict(cfg["model"]["fixed_params"].get(algorithm, {}))
    n_trials = cfg["model"]["n_hpo_trials"]
    early_stop = cfg["model"]["early_stopping_rounds"]
    metric_name = cfg["model"]["optimization_metric"]

    spw = float((y_in == 0).sum() / max((y_in == 1).sum(), 1)) if task_type == "binary" else 1.0
    rng = np.random.default_rng(seed)

    trials = []
    for _ in range(n_trials):
        params = sample_params(space, rng, spw, task_type)
        model = build_model(algorithm, params, fixed, seed, task_type)
        model = fit_model(algorithm, model, X_in, y_in, X_val, y_val, early_stop)

        if task_type == "regression":
            preds = model.predict(X_val)
            # HPO always keeps the highest val_score, so error metrics are negated
            rmse = mean_squared_error(y_val, preds) ** 0.5   # `squared=` was removed in newer sklearn
            score = (-rmse if metric_name == "rmse" else
                     -mean_absolute_error(y_val, preds) if metric_name == "mae" else
                     r2_score(y_val, preds))
        elif task_type == "multiclass":
            preds = model.predict(X_val)
            score = f1_score(y_val, preds, average="macro")   # "f1" is the only valid multiclass option
        else:  # binary
            proba = model.predict_proba(X_val)[:, 1]
            score = (f1_score(y_val, (proba >= 0.5).astype(int)) if metric_name == "f1" else
                     roc_auc_score(y_val, proba) if metric_name == "roc_auc" else
                     average_precision_score(y_val, proba))

        trials.append({**params, "best_iteration": best_iteration(algorithm, model),
                        "val_score": float(score)})

    res = pd.DataFrame(trials).sort_values("val_score", ascending=False).reset_index(drop=True)
    best = res.iloc[0].to_dict()

    best_params = {k: best[k] for k in space if k != "use_scale_pos_weight" and k in best}
    if "scale_pos_weight" in best:
        best_params["scale_pos_weight"] = best["scale_pos_weight"]
    for int_key in ("max_depth", "min_child_weight", "num_leaves", "min_child_samples", "n_estimators"):
        if int_key in best_params and best_params[int_key] is not None:
            best_params[int_key] = int(best_params[int_key])

    n_rounds = fixed.get("n_estimators")
    if algorithm in ("xgboost", "lightgbm") and best.get("best_iteration"):
        n_rounds = max(int(best["best_iteration"] * 1.1), 50)

    return best_params, n_rounds, res


def fit_final_model(cfg: dict, algorithm: str, best_params: dict, n_rounds, X_tr, y_tr, seed: int):
    fixed = dict(cfg["model"]["fixed_params"].get(algorithm, {}))
    if algorithm in ("xgboost", "lightgbm") and n_rounds:
        fixed["n_estimators"] = n_rounds
    model = build_model(algorithm, best_params, fixed, seed, cfg["model"].get("task_type", "binary"))
    model.fit(X_tr, y_tr)
    return model


# ============================================================================
# Metrics / risk bands
# ============================================================================
def score_predictions(y_true: np.ndarray, proba, yhat: np.ndarray, task_type: str = "binary") -> dict:
    """Metric set for a given (probability/confidence, hard-label) pair. Used
    directly by retraining's champion-vs-challenger comparison, where the
    champion's yhat already exists and shouldn't be re-derived from a new
    threshold.

    proba's meaning depends on task_type:
        binary      -> P(class == 1), one float per row
        multiclass  -> confidence = max class probability, one float per row
        regression  -> unused (pass None); yhat is the predicted value array
    """
    from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score,
                                  precision_score, recall_score, f1_score,
                                  mean_squared_error, mean_absolute_error, r2_score)

    if task_type == "regression":
        return {
            "rmse": float(mean_squared_error(y_true, yhat) ** 0.5),
            "mae": float(mean_absolute_error(y_true, yhat)),
            "r2": float(r2_score(y_true, yhat)),
        }

    k = max(int(0.10 * len(yhat)), 1)
    top = np.argsort(-proba)[:k]

    if task_type == "multiclass":
        return {
            "accuracy": float(accuracy_score(y_true, yhat)),
            "f1_macro": float(f1_score(y_true, yhat, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y_true, yhat, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, yhat, average="macro", zero_division=0)),
            # of the top-10% highest-confidence predictions, how many were actually correct
            "precision_at_decile": float((yhat[top] == y_true[top]).mean()),
        }

    # binary
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, yhat)),
        "precision": float(precision_score(y_true, yhat, zero_division=0)),
        "recall": float(recall_score(y_true, yhat, zero_division=0)),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        "recall_at_decile": float(y_true[top].sum() / max(y_true.sum(), 1)),
        "lift_at_decile": float(y_true[top].mean() / max(y_true.mean(), 1e-9)),
    }


def compute_metrics(y_true: np.ndarray, proba, threshold: float, task_type: str = "binary") -> dict:
    """Metric set derived from model output — used for a normal training
    run, where there's no pre-existing hard label.

    `proba`'s meaning depends on task_type:
        binary      -> P(class == 1) vector; thresholded here to get yhat
        multiclass  -> either the full (n, n_classes) probability matrix,
                       or an already-derived (n,) label array
        regression  -> the predicted value array (threshold is ignored)
    """
    if task_type == "regression":
        return score_predictions(y_true, None, proba, task_type)
    if task_type == "multiclass":
        yhat = proba.argmax(axis=1) if getattr(proba, "ndim", 1) == 2 else proba
        confidence = proba.max(axis=1) if getattr(proba, "ndim", 1) == 2 else None
        return score_predictions(y_true, confidence, yhat, task_type)
    yhat = (proba >= threshold).astype("int8")
    m = score_predictions(y_true, proba, yhat, task_type)
    m["base_rate"] = float(y_true.mean())
    m["threshold"] = float(threshold)
    return m


def evaluate_promotion(champion_metrics: dict, challenger_metrics: dict, promotion_cfg: dict,
                        task_type: str = "binary") -> dict:
    """Champion-vs-challenger promotion gate, config-driven instead of
    hardcoded thresholds.

    NOTE: promotion_cfg's keys (min_recall_at_decile_lift_pct,
    max_pr_auc_drop_pct) are named after the binary metrics they were
    originally written for. They're reused here as the generic "primary
    gate %" / "guardrail %" thresholds for all three task types — this
    works, but is worth renaming to task-neutral names (e.g.
    primary_metric_min_lift_pct / guardrail_max_drop_pct) next time
    config.yaml's schema is revised.
    """
    if task_type == "regression":
        # lower RMSE is better, so a "lift" here is a *reduction* in error
        rmse_delta_pct = ((champion_metrics["rmse"] - challenger_metrics["rmse"])
                           / max(champion_metrics["rmse"], 1e-9) * 100)
        r2_delta = challenger_metrics["r2"] - champion_metrics["r2"]
        promote = (rmse_delta_pct > promotion_cfg["min_recall_at_decile_lift_pct"]
                   and r2_delta > -(promotion_cfg["max_pr_auc_drop_pct"] / 100))
        return {"promote": bool(promote), "rmse_delta_pct": float(rmse_delta_pct), "r2_delta": float(r2_delta)}

    if task_type == "multiclass":
        f1_delta_pct = ((challenger_metrics["f1_macro"] - champion_metrics["f1_macro"])
                         / max(champion_metrics["f1_macro"], 1e-9) * 100)
        precision_delta_pct = ((challenger_metrics["precision_at_decile"] - champion_metrics["precision_at_decile"])
                                / max(champion_metrics["precision_at_decile"], 1e-9) * 100)
        promote = (f1_delta_pct > promotion_cfg["min_recall_at_decile_lift_pct"]
                   and precision_delta_pct > -promotion_cfg["max_pr_auc_drop_pct"])
        return {"promote": bool(promote), "f1_macro_delta_pct": float(f1_delta_pct),
                "precision_at_decile_delta_pct": float(precision_delta_pct)}

    # binary
    pr_auc_delta_pct = ((challenger_metrics["pr_auc"] - champion_metrics["pr_auc"])
                         / max(champion_metrics["pr_auc"], 1e-9) * 100)
    recall_delta_pct = ((challenger_metrics["recall_at_decile"] - champion_metrics["recall_at_decile"])
                         / max(champion_metrics["recall_at_decile"], 1e-9) * 100)
    roc_auc_delta = challenger_metrics["roc_auc"] - champion_metrics["roc_auc"]
    promote = (recall_delta_pct > promotion_cfg["min_recall_at_decile_lift_pct"]
               and pr_auc_delta_pct > -promotion_cfg["max_pr_auc_drop_pct"])
    return {
        "promote": bool(promote),
        "pr_auc_delta_pct": float(pr_auc_delta_pct),
        "recall_at_decile_delta_pct": float(recall_delta_pct),
        "roc_auc_delta": float(roc_auc_delta),
    }


def risk_cutoffs(proba: np.ndarray, n_high: int, n_medium: int):
    p_sorted = np.sort(proba)[::-1]
    n = len(p_sorted)
    cut_hi = float(p_sorted[min(n_high, n) - 1])
    cut_md = float(p_sorted[min(n_medium, n) - 1])
    return cut_hi, cut_md


def risk_band(proba: np.ndarray, cut_hi: float, cut_md: float):
    return np.where(proba >= cut_hi, "HIGH", np.where(proba >= cut_md, "MEDIUM", "LOW"))


# ============================================================================
# Explainability
# ============================================================================
def feature_contributions(algorithm, model, X: pd.DataFrame, feature_order, enc2base):
    """Per-row contributions, aggregated back to original (pre-one-hot)
    feature names. Uses SHAP when available; falls back to a rough
    importance x deviation approximation otherwise."""
    try:
        import shap
        explainer = (shap.LinearExplainer(model, X) if algorithm == "logistic_regression"
                     else shap.TreeExplainer(model))
        raw = explainer.shap_values(X)
        if isinstance(raw, list):
            raw = raw[1]
        raw = np.asarray(raw)
        if raw.ndim == 3:
            raw = raw[:, :, -1]
    except Exception as e:
        print(f"shap unavailable or failed ({e}); falling back to an importance-based approximation")
        raw = _fallback_contributions(model, X)

    contrib = pd.DataFrame(raw, columns=[enc2base[c] for c in feature_order])
    return contrib.T.groupby(level=0).sum().T


def _fallback_contributions(model, X: pd.DataFrame):
    if hasattr(model, "feature_importances_"):
        w = model.feature_importances_
    elif hasattr(model, "coef_"):
        w = model.coef_.ravel()
    else:
        w = np.ones(X.shape[1])
    centered = X.values - X.values.mean(axis=0)
    return centered * w


# ============================================================================
# Run / config registry  (the automated, no-hand-written-SQL layer)
# ============================================================================
def log_run(session, cfg: dict, resolved: dict, metrics: dict, extra: dict = None):
    """Writes one row per run to the shared config registry table. This is
    the only place SQL touches config, and it's generated, not typed."""
    payload = {
        "USE_CASE_NAME": cfg["use_case"]["name"],
        "PREFIX": cfg["use_case"]["prefix"],
        "MODEL_NAME": resolved["model_name"],
        "MODEL_VERSION": resolved["model_version"],
        "ALGORITHM": cfg["model"]["algorithm"],
        "CONFIG_JSON": json.dumps(_strip_meta(cfg)),
        "METRICS_JSON": json.dumps(metrics),
        "EXTRA_JSON": json.dumps(extra or {}, default=str),
        "CREATED_AT": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    df = pd.DataFrame([payload])
    session.write_pandas(
        df, resolved["config_table"],
        database=cfg["snowflake"]["database"], schema=cfg["snowflake"]["schema"],
        auto_create_table=True, overwrite=False, quote_identifiers=False,
    )


def load_past_run(session, cfg: dict, resolved: dict, model_version: str = None):
    """Reads a previous run back — the read-side counterpart to log_run(),
    so past runs (including the champion model a retrain challenges) are
    reproducible without SQL. Returns None if nothing matches, else:
        {"model_version", "algorithm", "created_at",
         "config":  the full config.yaml snapshot from that run,
         "metrics": that run's evaluation metrics,
         "extra":   best_params, n_estimators, impute_values, thresholds, ...}
    """
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}.{resolved["config_table"]}'
    t = session.table(fq)
    t = t.filter(t["USE_CASE_NAME"] == cfg["use_case"]["name"])
    if model_version:
        t = t.filter(t["MODEL_VERSION"] == model_version)
    row = t.sort(t["CREATED_AT"].desc()).limit(1).to_pandas()
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "model_version": r["MODEL_VERSION"],
        "algorithm": r["ALGORITHM"],
        "created_at": str(r["CREATED_AT"]),
        "config": json.loads(r["CONFIG_JSON"]),
        "metrics": json.loads(r["METRICS_JSON"]),
        "extra": json.loads(r["EXTRA_JSON"]) if r.get("EXTRA_JSON") else {},
    }


def log_feature_contract(session, cfg: dict, resolved: dict, feature_order, enc2base):
    contract = pd.DataFrame({
        "USE_CASE_NAME": cfg["use_case"]["name"],
        "MODEL_VERSION": resolved["model_version"],
        "ORDER_IDX": range(len(feature_order)),
        "ENCODED_NAME": feature_order,
        "SOURCE_NAME": [enc2base[c] for c in feature_order],
        "FEATURE_ROLE": ["CAT_OHE" if enc2base[c] in cfg["data"]["categorical_columns"] else "NUM"
                          for c in feature_order],
        "CREATED_AT": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    # one contract per (use case, version): replace any earlier batch instead of stacking duplicates
    contract_fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}.{resolved["contract_table"]}'
    uc_name, ver = cfg["use_case"]["name"], resolved["model_version"]
    try:
        session.sql(f"DELETE FROM {contract_fq} WHERE USE_CASE_NAME = '{uc_name}' "
                    f"AND MODEL_VERSION = '{ver}'").collect()
    except Exception:
        pass  # first ever run: the table does not exist yet, write_pandas creates it below
    session.write_pandas(
        contract, resolved["contract_table"],
        database=cfg["snowflake"]["database"], schema=cfg["snowflake"]["schema"],
        auto_create_table=True, overwrite=False, quote_identifiers=False,
    )


# ============================================================================
# Snowflake ML Observability — Model Monitor
# ============================================================================
def create_or_replace_model_monitor(session, cfg: dict, monitor_name: str, model_name: str,
                                     model_version: str, baseline_table: str, source_table: str,
                                     id_column: str):
    """Creates/replaces a Model Monitor watching `model_version` for drift,
    comparing `source_table` (live scored rows) against `baseline_table`.
    Requires CREATE MODEL MONITOR on the schema, SELECT on source_table,
    and USAGE on the warehouse + model (see setup_secrets.sql).

    Column contract written by every generic notebook, by task_type:
        binary      -> PREDICTED_PROBABILITY (float), PREDICTION (0/1), ACTUAL_LABEL (0/1)
        multiclass  -> PREDICTION (class label, cast to VARCHAR), ACTUAL_LABEL (class label, VARCHAR)
        regression  -> PREDICTED_VALUE (float), ACTUAL_LABEL (float)

    Per Snowflake's CREATE MODEL MONITOR rules: for multi-class models,
    predictions and actuals must both be class columns (STRING type); for
    regression models, both must be numeric. The notebook's baseline/monitor
    write step must cast PREDICTION/ACTUAL_LABEL to VARCHAR for multiclass
    runs accordingly.
    """
    mm_cfg = cfg.get("model_monitor", {})
    task_type = cfg["model"]["task_type"]
    db, schema, wh = cfg["snowflake"]["database"], cfg["snowflake"]["schema"], cfg["snowflake"]["warehouse"]

    if task_type == "regression":
        function_clause = "FUNCTION = PREDICT"
        score_block = "PREDICTION_SCORE_COLUMNS = ('PREDICTED_VALUE')"
        actual_block = "ACTUAL_SCORE_COLUMNS = ('ACTUAL_LABEL')"
    elif task_type == "multiclass":
        function_clause = "FUNCTION = PREDICT"
        score_block = "PREDICTION_CLASS_COLUMNS = ('PREDICTION')"
        actual_block = "ACTUAL_CLASS_COLUMNS = ('ACTUAL_LABEL')"
    else:  # binary — unchanged
        function_clause = "FUNCTION = PREDICT_PROBA"
        score_block = ("PREDICTION_SCORE_COLUMNS = ('PREDICTED_PROBABILITY')\n"
                        "        PREDICTION_CLASS_COLUMNS = ('PREDICTION')")
        actual_block = "ACTUAL_CLASS_COLUMNS = ('ACTUAL_LABEL')"

    sql = f"""
    CREATE OR REPLACE MODEL MONITOR {db}.{schema}.{monitor_name}
    WITH
        MODEL    = {db}.{schema}.{model_name}
        VERSION  = {model_version}
        {function_clause}
        WAREHOUSE = {wh}
        SOURCE   = {db}.{schema}.{source_table}
        BASELINE = {db}.{schema}.{baseline_table}
        TIMESTAMP_COLUMN = SCORED_AT
        ID_COLUMNS = ('{id_column}')
        {score_block}
        {actual_block}
        REFRESH_INTERVAL   = '{mm_cfg.get("refresh_interval", "1 minute")}'
        AGGREGATION_WINDOW = '{mm_cfg.get("aggregation_window", "1 day")}'
    """
    return session.sql(sql).collect()


# ============================================================================
# Inference deployment — input view + scoring stored procedure
#
# Everything below is read by generic_ml_inference_deploy_pipeline.ipynb
# only. It generates and deploys native Snowflake objects (a VIEW and a
# PROCEDURE) so that scoring itself runs entirely in-warehouse and can be
# triggered by anything that can issue a SQL CALL — including an externally
# managed orchestrator like Airflow. Nothing here runs at scoring time;
# these functions only run when you (re)deploy.
#
# Task-aware: binary, multiclass and regression are all supported here, driven
# by config.yaml's model.task_type (see probe_model_output_keys and
# build_scoring_procedure_sql).
# ============================================================================
MONITOR_FIXED_COLUMNS = ["PREDICTED_PROBABILITY", "PREDICTED_VALUE", "PREDICTION", "RISK_BAND",
                         "PROBABLE_CAUSES", "ACTUAL_LABEL", "SCORED_AT", "LABEL_MATURE_AT",
                         "MODEL_VERSION"]


def load_feature_contract(session, cfg: dict, resolved: dict, model_version: str) -> pd.DataFrame:
    """Feature contract rows (encoded name / source column / role) for one
    model version, straight from FEATURE_CONTRACT_REGISTRY — the source of
    truth for exactly which encoded columns a deployed model expects."""
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}.{resolved["contract_table"]}'
    t = session.table(fq)
    t = t.filter(t["USE_CASE_NAME"] == cfg["use_case"]["name"])
    t = t.filter(t["MODEL_VERSION"] == model_version)
    df = t.sort(t["ORDER_IDX"]).to_pandas()
    # a version trained more than once has several batches of rows here; every batch shares one
    # CREATED_AT, so the latest batch is the contract that belongs to the registered model
    if not df.empty:
        df = df[df["CREATED_AT"] == df["CREATED_AT"].max()].sort_values("ORDER_IDX").reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f'no feature contract found for {cfg["use_case"]["name"]} / {model_version} in '
            f'{resolved["contract_table"]} - run the training/retraining notebook for that version first.'
        )
    return df


def check_monitor_feature_parity(session, cfg: dict, resolved: dict, feature_order) -> dict:
    """Confirms the live monitor table's columns still match what
    `feature_order` expects. A mismatch usually means a retrain changed the
    feature set without the inference side being redeployed yet."""
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}'
    keys = cfg["data"]["key_columns"]
    cols = [r[0] for r in session.sql(f'DESC TABLE {fq}.{resolved["monitor_table"]}').collect()]
    actual_features = [c for c in cols if c not in keys and c not in MONITOR_FIXED_COLUMNS]
    extra = [c for c in actual_features if c not in feature_order]
    missing = [c for c in feature_order if c not in actual_features]
    return {
        "ok": not (extra or missing), "extra": extra, "missing": missing,
        "table_feature_count": len(actual_features), "model_feature_count": len(feature_order),
    }


def _normalize_sql_expr(col: str) -> str:
    """SQL equivalent of normalize_categorical(), inlined into generated
    view DDL so normalization runs in-warehouse at scoring time."""
    return (f"COALESCE(NULLIF(TRIM(REGEXP_REPLACE(UPPER(TRIM(TO_VARCHAR({col}))), "
            f"'[^A-Z0-9]+', '_'), '_'), ''), 'MISSING')")


def build_scoring_input_view_sql(cfg: dict, resolved: dict, feature_contract: pd.DataFrame,
                                  source_view: str, passthrough_cols, impute_values: dict) -> str:
    """Generates the CREATE VIEW DDL that turns a raw streaming/batch source
    into exactly the encoded columns the registered model expects — the
    same normalization + one-hot logic as encode_features(), expressed in
    SQL so it runs entirely in-warehouse at scoring time."""
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}'
    cat_cols = cfg["data"]["categorical_columns"]
    num_cols = cfg["data"]["numeric_columns"]
    role = dict(zip(feature_contract["ENCODED_NAME"], feature_contract["FEATURE_ROLE"]))
    src_of = dict(zip(feature_contract["ENCODED_NAME"], feature_contract["SOURCE_NAME"]))
    feature_order = list(feature_contract["ENCODED_NAME"])

    # fail with a readable message instead of a Snowflake "duplicate column name" compile error
    from collections import Counter
    def _dupes(names):
        return sorted(n for n, c in Counter(str(x).upper() for x in names).items() if c > 1)
    dupes = sorted(set(
        _dupes(list(passthrough_cols) + feature_order)
        + _dupes(list(passthrough_cols) + [f"N_{c}" for c in cat_cols] + list(num_cols))
    ))
    if dupes:
        raise ValueError(
            f"input view would have duplicate column(s) {dupes}. Check: (1) the feature contract table "
            f"for stacked rows of this model version (each training run used to append a full copy), "
            f"(2) inference.passthrough_columns overlapping a feature, "
            f"(3) a column listed twice under data.numeric_columns."
        )

    src_lines = [f"    {passthrough_cols[0]}"] + [f"  , {c}" for c in passthrough_cols[1:]]
    src_lines += [f"  , {_normalize_sql_expr(c)} AS N_{c}" for c in cat_cols]
    src_lines += [f"  , COALESCE({c}::FLOAT, {impute_values[c]}) AS {c}" if c in impute_values
                  else f"  , {c}::FLOAT AS {c}" for c in num_cols]

    out_lines = [f"    {passthrough_cols[0]}"] + [f"  , {c}" for c in passthrough_cols[1:]]
    for enc in feature_order:
        if role[enc] == "NUM":
            out_lines.append(f"  , {enc}")
        else:
            base = src_of[enc]
            val = enc[len(base) + 1:]
            out_lines.append(f"  , IFF(N_{base} = '{val}', 1, 0)::FLOAT AS {enc}")

    return (f'CREATE OR REPLACE VIEW {fq}.{resolved["input_view"]} AS\n'
            f"WITH src AS (\n  SELECT\n" + "\n".join(src_lines) +
            f"\n  FROM {fq}.{source_view}\n)\nSELECT\n" + "\n".join(out_lines) + "\nFROM src")


def deploy_scoring_input_view(session, cfg: dict, resolved: dict, feature_contract: pd.DataFrame,
                               source_view: str, passthrough_cols, impute_values: dict):
    """Builds and executes the input view DDL, then verifies every expected
    feature column actually came out the other end."""
    # Pre-flight: the view DDL selects these by unquoted name, so each must exist in the source as an
    # upper-case column. Snowflake reports only the first unknown column per attempt, so check them all here.
    fq_src = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}'
    src_cols = list(session.table(f"{fq_src}.{source_view}").columns)
    plain = {c.upper() for c in src_cols if not c.startswith('"')}
    quoted = {c.strip('"').upper(): c for c in src_cols if c.startswith('"')}
    needed = list(dict.fromkeys(list(passthrough_cols) + list(cfg["data"]["categorical_columns"])
                                + list(cfg["data"]["numeric_columns"])))
    absent = [c for c in needed if c.upper() not in plain]
    if absent:
        case_hint = {c: quoted[c.upper()] for c in absent if c.upper() in quoted}
        raise ValueError(
            f"{source_view} is missing {len(absent)} column(s) that the scoring view selects: {absent[:20]}"
            + (f" (present only with different letter case: {case_hint})" if case_hint else "")
            + f". Columns it does have: {src_cols[:40]}. Fix inference.source_stream_view, or remove those "
              f"columns from inference.passthrough_columns / data.categorical_columns / data.numeric_columns."
        )

    ddl = build_scoring_input_view_sql(cfg, resolved, feature_contract, source_view,
                                        passthrough_cols, impute_values)
    session.sql(ddl).collect()
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}'
    cols = [r[0] for r in session.sql(f'DESC VIEW {fq}.{resolved["input_view"]}').collect()]
    feature_order = list(feature_contract["ENCODED_NAME"])
    missing = [c for c in feature_order if c not in cols]
    if missing:
        raise ValueError(f'columns missing from {resolved["input_view"]}: {missing[:5]}')
    wm_col = (cfg.get("inference") or {}).get("watermark_column")
    if wm_col and wm_col.upper() not in {c.upper() for c in cols}:
        raise ValueError(f'inference.watermark_column {wm_col} is not in {resolved["input_view"]}: add it to '
                         f'inference.passthrough_columns, since the scoring procedure filters on it.')
    return cols


def probe_predict_proba_key(session, cfg: dict, resolved: dict, model_name: str,
                             model_version: str, feature_order) -> str:
    """PREDICT_PROBA's output key naming (output_feature_N) isn't something
    to hardcode — probe it once against the real deployed version instead."""
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}'
    args = ", ".join(feature_order)
    p = session.sql(
        f'WITH m AS MODEL {fq}.{model_name} VERSION {model_version} '
        f'SELECT m!PREDICT_PROBA({args}) AS P FROM {fq}.{resolved["input_view"]} LIMIT 1'
    ).collect()
    if not p:
        return "output_feature_1"
    keys = list(json.loads(p[0]["P"]).keys())
    positive_keys = [k for k in keys if k.endswith("1")]
    return positive_keys[0] if positive_keys else keys[-1]


def probe_model_output_keys(session, cfg: dict, resolved: dict, model_name: str,
                             model_version: str, feature_order) -> list:
    """Task-aware successor to probe_predict_proba_key(). Calls the deployed
    model version once and reads the keys of the output object it returns,
    instead of hardcoding Snowflake's `output_feature_N` naming:

        binary      -> PREDICT_PROBA -> [key of the positive class]
        multiclass  -> PREDICT_PROBA -> [one key per class]
        regression  -> PREDICT       -> [the single value key]
    """
    task_type = cfg["model"]["task_type"]
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}'
    fn = "PREDICT" if task_type == "regression" else "PREDICT_PROBA"
    args = ", ".join(feature_order)
    p = session.sql(
        f'WITH m AS MODEL {fq}.{model_name} VERSION {model_version} '
        f'SELECT m!{fn}({args}) AS P FROM {fq}.{resolved["input_view"]} LIMIT 1'
    ).collect()
    if not p:
        if task_type == "binary":
            return ["output_feature_1"]   # unchanged legacy fallback
        raise ValueError(
            f'{resolved["input_view"]} returned no rows, so the {fn} output keys cannot be probed. '
            f'The source stream view needs at least one row at deploy time.'
        )
    keys = list(json.loads(p[0]["P"]).keys())
    if task_type == "binary":
        positive_keys = [k for k in keys if k.endswith("1")]
        return [positive_keys[0] if positive_keys else keys[-1]]
    if task_type == "multiclass":
        return sorted(keys)
    return [keys[0]]


def probe_explain_suffix(session, cfg: dict, resolved: dict, model_name: str, model_version: str,
                          feature_order):
    """Probes whether EXPLAIN is available for this model version and, if
    so, what suffix it appends to each feature's explanation column.
    Returns (has_explain: bool, suffix: str). Never raises — falls back to
    (False, "_EXPLANATION") so a deploy can proceed with PROBABLE_CAUSES
    simply left NULL rather than blocking on it."""
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}'
    from snowflake.ml.registry import Registry
    mv = (Registry(session, database_name=cfg["snowflake"]["database"], schema_name=cfg["snowflake"]["schema"])
          .get_model(model_name).version(model_version))
    fns = [f["name"].upper() for f in mv.show_functions()]
    if "EXPLAIN" not in fns:
        return False, "_EXPLANATION"
    probe_table = f'{resolved["input_view"]}_XAI_PROBE'
    try:
        args_v = ", ".join(f"v.{c}" for c in feature_order)
        key_col = cfg["data"]["key_columns"][0]
        session.sql(f'CREATE OR REPLACE TRANSIENT TABLE {fq}.{probe_table} AS '
                    f'SELECT * FROM {fq}.{resolved["input_view"]} LIMIT 3').collect()
        cols = session.sql(
            f'WITH m AS MODEL {fq}.{model_name} VERSION {model_version} '
            f'SELECT v.{key_col}, e.* FROM {fq}.{probe_table} v, '
            f'TABLE(m!EXPLAIN({args_v}) OVER (PARTITION BY v.{key_col})) e'
        ).to_pandas().columns.tolist()
        suffix = next(c[len(f):] for f in feature_order for c in cols
                      if c.upper() == (f + "_EXPLANATION").upper())
        return True, suffix
    except Exception as e:
        print(f"EXPLAIN probe failed, disabling PROBABLE_CAUSES: {e}")
        return False, "_EXPLANATION"
    finally:
        session.sql(f"DROP TABLE IF EXISTS {fq}.{probe_table}").collect()


def build_scoring_procedure_sql(cfg: dict, resolved: dict, model_name: str, model_version: str,
                                 feature_order, proba_key: str, has_explain: bool, expl_suffix: str,
                                 threshold: float, cut_hi: float, cut_md: float, horizon_days: int,
                                 watermark_col: str, xai_bucket_count: int, output_keys: list = None) -> str:
    """Generates the CREATE PROCEDURE DDL for daily scoring: reads new rows
    from the input view since the last watermark, scores + (optionally)
    explains them in-warehouse, writes the monitor table + an optional
    use-case business table inside one transaction, advances the
    watermark, and logs the run — all as native Snowflake SQL, callable
    with a single `CALL` by any external orchestrator (e.g. Airflow).

    Created EXECUTE AS OWNER (Snowflake's default): the caller therefore
    only ever needs USAGE on the procedure itself, never direct access to
    the tables it reads or writes.

    Task-aware (config.yaml: model.task_type). What lands in the monitor
    table matches the column contract training's baseline step writes:

        binary      PREDICTED_PROBABILITY = P(class 1); PREDICTION = 0/1 vs `threshold`;
                    RISK_BAND from cut_hi / cut_md
        multiclass  PREDICTED_PROBABILITY = max class probability (confidence);
                    PREDICTION = predicted class as VARCHAR; RISK_BAND on confidence
        regression  PREDICTED_VALUE, and PREDICTION carrying the same number; RISK_BAND NULL

    `output_keys` are the keys of the model function's output object, from
    probe_model_output_keys(): one key for binary/regression, one per class
    for multiclass. `proba_key` is kept so existing binary callers still work.
    For multiclass, each class's label is taken from the suffix of its key
    (output_feature_2 -> '2'), which matches the integer class codes training
    label-encodes the target to.
    """
    task_type = cfg["model"]["task_type"]
    out_keys = list(output_keys) if output_keys else [proba_key]
    fq = f'{cfg["snowflake"]["database"]}.{cfg["snowflake"]["schema"]}'
    keys = cfg["data"]["key_columns"]
    key = keys[0]
    args = ", ".join(feature_order)
    proc_name = resolved["procedure_name"]
    input_view = resolved["input_view"]
    mon_table = resolved["monitor_table"]
    log_table = resolved["log_table"]
    wm_table = resolved["watermark_table"]
    use_case = cfg["use_case"]["name"]
    score_raw = f"{proc_name}_SCORE_RAW"
    expl_raw = f"{proc_name}_EXPL_RAW"
    top_n_causes = cfg["explainability"]["top_n_causes_per_row"]

    # ---- per-task pieces: model function, scoring CTEs, monitor columns ----
    model_fn = "PREDICT" if task_type == "regression" else "PREDICT_PROBA"
    if task_type in ("binary", "multiclass"):
        assert cut_hi is not None and cut_md is not None, \
            "risk band cutoffs are required for binary/multiclass deployments"

    if task_type == "binary":
        score_ctes = f"""scored AS (
        SELECT r.*, GET(r.PRED_OBJ, '{out_keys[0]}')::FLOAT AS PROB
        FROM {fq}.{score_raw} r
    )"""
        insert_score_cols = "PREDICTED_PROBABILITY, PREDICTION, RISK_BAND"
        score_select = f"""s.PROB                                       AS PREDICTED_PROBABILITY,
           IFF(s.PROB >= {threshold}, 1, 0)::NUMBER(1,0) AS PREDICTION,
           CASE WHEN s.PROB >= {cut_hi} THEN 'HIGH'
                WHEN s.PROB >= {cut_md} THEN 'MEDIUM'
                ELSE 'LOW' END                          AS RISK_BAND"""
        actual_null = "NULL::NUMBER(1,0)"
        biz_score_cols = ["PREDICTED_PROBABILITY", "RISK_BAND", "PREDICTION"]

    elif task_type == "multiclass":
        assert len(out_keys) >= 2, "multiclass needs one PREDICT_PROBA output key per class"
        labels = [k.rsplit("_", 1)[-1] for k in out_keys]
        p_cols = ", ".join(f"GET(r.PRED_OBJ, '{k}')::FLOAT AS P_{i}" for i, k in enumerate(out_keys))
        p_names = ", ".join(f"P_{i}" for i in range(len(out_keys)))
        pick = " ".join(f"WHEN P_{i} = GREATEST({p_names}) THEN '{lab}'"
                        for i, lab in enumerate(labels[:-1]))
        score_ctes = f"""scored0 AS (
        SELECT r.*, {p_cols}
        FROM {fq}.{score_raw} r
    ), scored AS (
        SELECT s0.*,
               GREATEST({p_names}) AS PROB,
               CASE {pick} ELSE '{labels[-1]}' END AS PRED_LABEL
        FROM scored0 s0
    )"""
        insert_score_cols = "PREDICTED_PROBABILITY, PREDICTION, RISK_BAND"
        score_select = f"""s.PROB                                       AS PREDICTED_PROBABILITY,
           s.PRED_LABEL::VARCHAR                        AS PREDICTION,
           CASE WHEN s.PROB >= {cut_hi} THEN 'HIGH'
                WHEN s.PROB >= {cut_md} THEN 'MEDIUM'
                ELSE 'LOW' END                          AS RISK_BAND"""
        actual_null = "NULL::VARCHAR"
        biz_score_cols = ["PREDICTED_PROBABILITY", "RISK_BAND", "PREDICTION"]

    else:  # regression
        score_ctes = f"""scored AS (
        SELECT r.*, GET(r.PRED_OBJ, '{out_keys[0]}')::FLOAT AS PRED_VALUE
        FROM {fq}.{score_raw} r
    )"""
        insert_score_cols = "PREDICTED_VALUE, PREDICTION, RISK_BAND"
        score_select = """s.PRED_VALUE                                AS PREDICTED_VALUE,
           s.PRED_VALUE                                 AS PREDICTION,
           NULL::VARCHAR                                AS RISK_BAND"""
        actual_null = "NULL::FLOAT"
        biz_score_cols = ["PREDICTED_VALUE", "PREDICTION"]

    if has_explain:
        args_r = ", ".join(f"r.{c}" for c in feature_order)
        expl_table_sql = f"""
    CREATE OR REPLACE TRANSIENT TABLE {fq}.{expl_raw} AS
    WITH m AS MODEL {fq}.{model_name} VERSION {model_version}
    SELECT r.{key}, e.*
    FROM {fq}.{score_raw} r,
         TABLE(m!EXPLAIN({args_r}) OVER (PARTITION BY r.XAI_BUCKET)) e;
"""
        if task_type == "multiclass":
            # EXPLAIN returns one SHAP value per class for every feature, as JSON text keyed by
            # class label (e.g. '{"0": 1.37, "1": -0.64, "2": -0.71}'), so parse it and read the
            # value for the class the model actually predicted for that row. Going through
            # VARCHAR + TRY_PARSE_JSON handles an object, a JSON string, and a plain number alike.
            expl_src = f"""(
            SELECT e.{key} AS {key}, sc.PRED_LABEL AS PRED_LABEL, e.OBJ AS OBJ
            FROM (SELECT {key}, OBJECT_CONSTRUCT(*) AS OBJ FROM {fq}.{expl_raw}) e
            JOIN scored sc ON sc.{key} = e.{key}
        ) x"""
            pj = "TRY_PARSE_JSON(f.value::VARCHAR)"
            contrib = f"TRY_TO_DOUBLE(IFF(IS_OBJECT({pj}), GET({pj}, x.PRED_LABEL), {pj})::VARCHAR)"
        else:
            expl_src = f"(SELECT {key}, OBJECT_CONSTRUCT(*) AS OBJ FROM {fq}.{expl_raw}) x"
            # The flattened object also holds the key column (a text value such as a policy number), and
            # Snowflake may evaluate this expression before the join to the contract discards that row.
            # TRY_TO_DOUBLE turns anything non-numeric into NULL instead of failing the whole call.
            contrib = "TRY_TO_DOUBLE(f.value::VARCHAR)"
        causes_cte = f""",
    causes AS (
        SELECT {key},
               ARRAY_TO_STRING(ARRAY_SLICE(ARRAY_AGG(BASE)
                   WITHIN GROUP (ORDER BY CONTRIB DESC), 0, {top_n_causes}), ' | ') AS PROBABLE_CAUSES
        FROM (
            SELECT x.{key}, c.SOURCE_NAME AS BASE, SUM({contrib}) AS CONTRIB
            FROM {expl_src},
                 LATERAL FLATTEN(input => x.OBJ) f
            JOIN {fq}.{resolved["contract_table"]} c
              ON UPPER(c.ENCODED_NAME) = UPPER(REPLACE(f.key, '{expl_suffix}', ''))
             AND c.MODEL_VERSION = '{model_version}' AND c.USE_CASE_NAME = '{use_case}'
            GROUP BY 1, 2
            HAVING SUM({contrib}) > 0
        )
        GROUP BY 1
    )"""
        causes_col = "c.PROBABLE_CAUSES"
        causes_join = f"LEFT JOIN causes c ON s.{key} = c.{key}"
    else:
        expl_table_sql, causes_cte, causes_join = "", "", ""
        causes_col = "NULL::VARCHAR AS PROBABLE_CAUSES"

    all_keys = ", ".join(keys)
    all_keys_s = ", ".join(f"s.{k}" for k in keys)

    inf = cfg.get("inference", {}) or {}
    biz_table = inf.get("business_table")
    biz_map = inf.get("business_table_column_map") or {}
    if biz_table and biz_map:
        biz_cols_out = list(biz_map.values())
        merge_key = biz_cols_out[0]
        biz_select_src = ", ".join(f"r.{src} AS {dst}" for src, dst in biz_map.items())
        biz_model_cols = biz_score_cols + ["PROBABLE_CAUSES"]
        biz_update = ",\n        ".join(
            [f"t.{c} = s.{c}" for c in biz_cols_out[1:]]
            + [f"t.{c} = s.{c}" for c in biz_model_cols]
            + ["t.SYS_UPD_DT = :run_ts"]
        )
        biz_merge_sql = f"""
    MERGE INTO {fq}.{biz_table} t
    USING (
        SELECT {", ".join(biz_cols_out)},
               {", ".join(biz_model_cols)}
        FROM (
            SELECT {biz_select_src},
                   {", ".join("m." + c for c in biz_model_cols)},
                   ROW_NUMBER() OVER (PARTITION BY m.{key} ORDER BY m.SCORED_AT DESC) rn
            FROM {fq}.{mon_table} m
            JOIN {fq}.{score_raw} r ON r.{key} = m.{key}
            WHERE m.SCORED_AT = :run_ts
        ) WHERE rn = 1
    ) s ON t.{merge_key} = s.{merge_key}
    WHEN MATCHED THEN UPDATE SET
        {biz_update}
    WHEN NOT MATCHED THEN INSERT
        ({", ".join(biz_cols_out)}, {", ".join(biz_model_cols)}, SYS_UPD_DT)
        VALUES ({", ".join(f"s.{c}" for c in biz_cols_out)},
                {", ".join(f"s.{c}" for c in biz_model_cols)}, :run_ts);
"""
    else:
        biz_merge_sql = ""

    return f"""
CREATE OR REPLACE PROCEDURE {fq}.{proc_name}()
RETURNS STRING LANGUAGE SQL
EXECUTE AS OWNER
AS
$$
DECLARE
    run_ts TIMESTAMP_NTZ(6);
    cutoff TIMESTAMP_NTZ(6);
    wm     TIMESTAMP_NTZ(6);
    new_wm TIMESTAMP_NTZ(6);
    n_row  NUMBER DEFAULT 0;
BEGIN
    run_ts := CURRENT_TIMESTAMP()::TIMESTAMP_NTZ(6);
    cutoff := CURRENT_TIMESTAMP()::DATE::TIMESTAMP_NTZ(6);
    SELECT MAX(LAST_RECORD_TS) INTO wm FROM {fq}.{wm_table} WHERE USE_CASE_NAME = '{use_case}';
    IF (wm IS NULL) THEN
        -- first ever run for this use case: no watermark row yet, so start from the beginning
        wm := '1900-01-01'::TIMESTAMP_NTZ(6);
    END IF;

    CREATE OR REPLACE TRANSIENT TABLE {fq}.{score_raw} AS
    WITH m AS MODEL {fq}.{model_name} VERSION {model_version}
    SELECT v.*,
           m!{model_fn}({args}) AS PRED_OBJ,
           MOD(ABS(HASH(v.{key})), {xai_bucket_count}) AS XAI_BUCKET
    FROM {fq}.{input_view} v
    WHERE v.{watermark_col} > :wm AND v.{watermark_col} <= :cutoff;

    SELECT COUNT(*), MAX({watermark_col}) INTO n_row, new_wm FROM {fq}.{score_raw};

    IF (n_row = 0) THEN
        INSERT INTO {fq}.{log_table}
        VALUES (:run_ts, '{use_case}', '{proc_name}', 0, 'SKIPPED', 'no new rows to score');
        RETURN 'SKIPPED: 0 new rows';
    END IF;
{expl_table_sql}
    BEGIN TRANSACTION;

    INSERT INTO {fq}.{mon_table}
        ({all_keys}, {args},
         {insert_score_cols}, PROBABLE_CAUSES,
         ACTUAL_LABEL, SCORED_AT, LABEL_MATURE_AT, MODEL_VERSION)
    WITH {score_ctes}{causes_cte}
    SELECT {all_keys_s},
           {args},
           {score_select},
           {causes_col},
           {actual_null}                                AS ACTUAL_LABEL,
           :run_ts                                      AS SCORED_AT,
           DATEADD(day, {horizon_days}, :run_ts)::TIMESTAMP_NTZ(6) AS LABEL_MATURE_AT,
           '{model_version}'                            AS MODEL_VERSION
    FROM scored s {causes_join};
{biz_merge_sql}
    MERGE INTO {fq}.{wm_table} t USING (SELECT '{use_case}' AS USE_CASE_NAME) s
    ON t.USE_CASE_NAME = s.USE_CASE_NAME
    WHEN MATCHED THEN UPDATE SET t.LAST_RECORD_TS = :new_wm, t.UPDATED_AT = :run_ts
    WHEN NOT MATCHED THEN INSERT (USE_CASE_NAME, LAST_RECORD_TS, UPDATED_AT)
        VALUES ('{use_case}', :new_wm, :run_ts);

    COMMIT;

    INSERT INTO {fq}.{log_table}
    VALUES (:run_ts, '{use_case}', '{proc_name}', :n_row, 'OK', 'scored ' || :n_row || ' rows');

    RETURN 'OK: ' || :n_row || ' rows scored';

EXCEPTION
    WHEN OTHER THEN
        ROLLBACK;
        BEGIN
            INSERT INTO {fq}.{log_table}
            VALUES (:run_ts, '{use_case}', '{proc_name}', 0, 'FAILED', :sqlerrm);
        EXCEPTION
            WHEN OTHER THEN NULL;
        END;
        RAISE;
END;
$$
"""


def deploy_scoring_procedure(session, cfg: dict, resolved: dict, **kwargs) -> str:
    """Builds the scoring procedure DDL (see build_scoring_procedure_sql for
    kwargs) and executes it. Returns the DDL text for inspection/logging."""
    ddl = build_scoring_procedure_sql(cfg, resolved, **kwargs)
    session.sql(ddl).collect()
    return ddl
