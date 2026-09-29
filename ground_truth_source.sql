-- =============================================================================
-- ground_truth_source.sql  (USE-CASE-OWNED - Complaint Propensity)
--
-- This is the ONLY SQL a use case owns for retraining. It defines the
-- ground-truth source named by config.yaml -> retraining.ground_truth_source:
-- a table or view with one row per policy that says what actually happened.
--
--   contract:  POLICY_NO (the key)  +  the outcome column named in
--              retraining.ground_truth_label_column
--
-- Everything else is generic and lives in the shared engine
-- (ml_pipeline_utils.py): merging these outcomes into the monitor table and
-- assembling the retraining dataset. That is why the old ground_truth_merge.sql
-- and retrain_base_view.sql are gone.
--
-- Placeholders in curly braces are filled from config.yaml by the notebook, so
-- nothing here hardcodes a database, schema or table name.
--
-- ---------------------------------------------------------------------------
-- SANDBOX ONLY - SYNTHETIC LABELS
-- There is no real complaints feed in the sandbox, so outcomes are simulated:
-- a weighted ranking of a few features plus deterministic noise, with the top
-- 0.18 percent flagged as complaints. Retraining on these labels only proves the
-- pipeline works end to end. A model trained on them has learned a heuristic,
-- not real complaint behaviour, and must never be promoted for production use.
-- For a real use case, replace this file with a view over the actual complaints
-- system, and set retraining.synthetic_labels to false.
--
-- Two differences from the earlier version, both on purpose:
--   * it is a view, not a table rebuilt on every run, so labels can no longer
--     disappear from one run to the next
--   * it ranks all policies in the feed, not just the currently unlabeled ones,
--     so a policy's label does not depend on which batch it arrived in
-- =============================================================================
CREATE OR REPLACE VIEW {FQ}.{GT_SOURCE} AS
WITH latest AS (
    SELECT POLICY_NO,
           CASE_TOTAL_CASES_CNT, CALL_ACTIVITY_TOTAL_CNT, ORPHAN_FLAG,
           PERCENTAGE_PREMI_INCOME, LOSS_GAIN_INVESTMENT, TRAN_PREMIUM, CUST_VALUE_ENCODED
    FROM {FQ}.{PROD_FEED}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY POLICY_NO ORDER BY {WATERMARK_COL} DESC) = 1
),
ranked AS (
    SELECT POLICY_NO,
           PERCENT_RANK() OVER (ORDER BY COALESCE(CASE_TOTAL_CASES_CNT, 0))    AS R_CASE,
           PERCENT_RANK() OVER (ORDER BY COALESCE(CALL_ACTIVITY_TOTAL_CNT, 0)) AS R_CALL,
           PERCENT_RANK() OVER (ORDER BY COALESCE(PERCENTAGE_PREMI_INCOME, 0)) AS R_PREMI,
           PERCENT_RANK() OVER (ORDER BY COALESCE(TRAN_PREMIUM, 0))            AS R_TRAN,
           PERCENT_RANK() OVER (ORDER BY COALESCE(CUST_VALUE_ENCODED, 0))      AS R_VALUE,
           COALESCE(ORPHAN_FLAG, 0)                                            AS F_ORPHAN,
           IFF(COALESCE(LOSS_GAIN_INVESTMENT, 0) < 0, 1, 0)                    AS F_RUGI
    FROM latest
),
scored AS (
    SELECT POLICY_NO,
             1.00 * R_CASE
           + 0.90 * F_ORPHAN
           + 0.70 * R_CALL
           + 0.40 * R_PREMI
           + 0.35 * F_RUGI
           + 0.25 * R_TRAN
           - 0.30 * R_VALUE
           + 1.20 * ((ABS(HASH(POLICY_NO)) % 100000) / 100000.0)
           AS SKOR
    FROM ranked
)
SELECT POLICY_NO,
       IFF(PERCENT_RANK() OVER (ORDER BY SKOR DESC) < 0.0018, 1, 0)::NUMBER(1,0) AS COMPLAIN_FLAG,
       ROUND(SKOR, 4) AS SKOR_SINTETIS
FROM scored
