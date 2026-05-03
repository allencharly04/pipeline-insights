"""
Pipeline Pulse — analytics layer.

Centralizes all KPI computation logic so the Streamlit dashboard, Power BI
report, and Groq agent all reference the same definitions.

Each function takes the SQLite database path (or uses the default) and returns
a pandas DataFrame ready for plotting or summarization.

Definitions used throughout:
    - "Won deals":         current_stage = 'Closed Won'
    - "Lost deals":        current_stage = 'Closed Lost'
    - "Closed deals":      either of the above
    - "Active pipeline":   not closed (i.e. still in Prospecting/Qualified/Proposal/Negotiation)
    - "Win rate":          won / (won + lost)  — i.e. of CLOSED deals, what fraction won
                           (open deals deliberately excluded since they haven't decided yet)
    - "Pipeline value":    sum of amount_eur for active deals
    - "Weighted pipeline": pipeline value × stage win-probability (the forecast)
"""

from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

# Default DB path — relative to the project root.
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "pipeline.db"

# Stage win-probabilities — used for weighted pipeline forecasting.
# These are the standard probabilities a sales ops team would set, calibrated
# to the actual conversion rates in our generated data.
STAGE_WIN_PROBABILITY = {
    "Prospecting": 0.10,
    "Qualified": 0.25,
    "Proposal": 0.50,
    "Negotiation": 0.75,
    "Closed Won": 1.00,
    "Closed Lost": 0.00,
}


def get_engine(db_path=None):
    """Return a SQLAlchemy engine for the pipeline DB."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    return create_engine(f"sqlite:///{path}")


# ---------------------------------------------------------------------------
# Top-level summary KPIs
# ---------------------------------------------------------------------------

def pipeline_summary(db_path=None):
    """
    Returns a one-row DataFrame with the headline KPIs:
        - total_opportunities
        - active_pipeline_value (open deals)
        - won_value (closed-won total revenue)
        - won_count, lost_count
        - win_rate (decimal, e.g. 0.28 = 28%)
        - avg_won_deal_size
        - avg_days_to_close (won deals only)
    """
    engine = get_engine(db_path)
    query = """
        SELECT
            COUNT(*) AS total_opportunities,
            SUM(CASE WHEN is_closed = 0 THEN amount_eur ELSE 0 END) AS active_pipeline_value,
            SUM(CASE WHEN is_won = 1 THEN amount_eur ELSE 0 END) AS won_value,
            SUM(CASE WHEN is_won = 1 THEN 1 ELSE 0 END) AS won_count,
            SUM(CASE WHEN current_stage = 'Closed Lost' THEN 1 ELSE 0 END) AS lost_count,
            AVG(CASE WHEN is_won = 1 THEN amount_eur END) AS avg_won_deal_size,
            AVG(CASE WHEN is_won = 1 THEN days_open END) AS avg_days_to_close
        FROM opportunities
    """
    df = pd.read_sql(query, engine)
    # Win rate: of closed deals, what fraction won
    closed = df.loc[0, "won_count"] + df.loc[0, "lost_count"]
    df["win_rate"] = df.loc[0, "won_count"] / closed if closed > 0 else 0.0
    return df


# ---------------------------------------------------------------------------
# Funnel / stage analysis
# ---------------------------------------------------------------------------

def pipeline_by_stage(db_path=None):
    """
    Returns one row per stage with deal count and total value.
    Stages returned in standard funnel order.
    """
    engine = get_engine(db_path)
    query = """
        SELECT
            current_stage AS stage,
            COUNT(*) AS deal_count,
            ROUND(SUM(amount_eur), 2) AS total_value_eur,
            ROUND(AVG(amount_eur), 2) AS avg_deal_size_eur
        FROM opportunities
        GROUP BY current_stage
    """
    df = pd.read_sql(query, engine)
    # Force standard order
    stage_order = ["Prospecting", "Qualified", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
    df["stage"] = pd.Categorical(df["stage"], categories=stage_order, ordered=True)
    return df.sort_values("stage").reset_index(drop=True)


def stage_conversion_rates(db_path=None):
    """
    Computes the conversion rate from each stage to the next, using stage_history.
    For each opportunity, find the earliest entry into each stage and check if
    it ever advanced.
    """
    engine = get_engine(db_path)
    history = pd.read_sql("SELECT * FROM stage_history", engine)
    stages = ["Prospecting", "Qualified", "Proposal", "Negotiation"]
    rows = []
    for i, stage in enumerate(stages):
        next_stages = stages[i + 1:] + ["Closed Won"]
        # opps that ever entered this stage
        entered = set(history.loc[history["stage"] == stage, "opportunity_id"])
        # of those, how many ever entered any later stage (advanced) vs. went to Closed Lost
        advanced = set(history.loc[history["stage"].isin(next_stages), "opportunity_id"]) & entered
        conv = len(advanced) / len(entered) if entered else 0.0
        rows.append({
            "from_stage": stage,
            "entered_count": len(entered),
            "advanced_count": len(advanced),
            "conversion_rate": round(conv, 3),
        })
    return pd.DataFrame(rows)


def avg_days_in_stage(db_path=None):
    """
    Average number of days deals spend at each stage before moving on.
    Only computes for stages where the deal HAS moved on (i.e. we know the duration).
    """
    engine = get_engine(db_path)
    history = pd.read_sql(
        "SELECT * FROM stage_history ORDER BY opportunity_id, stage_order", engine
    )
    history["entered_at"] = pd.to_datetime(history["entered_at"])
    history["next_entered_at"] = history.groupby("opportunity_id")["entered_at"].shift(-1)
    history["days_in_stage"] = (history["next_entered_at"] - history["entered_at"]).dt.days
    completed = history.dropna(subset=["days_in_stage"])
    avg = (
        completed[completed["stage"].isin(["Prospecting", "Qualified", "Proposal", "Negotiation"])]
        .groupby("stage")["days_in_stage"]
        .agg(["mean", "median", "count"])
        .round(1)
        .reset_index()
        .rename(columns={"mean": "avg_days", "median": "median_days", "count": "n_transitions"})
    )
    stage_order = ["Prospecting", "Qualified", "Proposal", "Negotiation"]
    avg["stage"] = pd.Categorical(avg["stage"], categories=stage_order, ordered=True)
    return avg.sort_values("stage").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Segmentation: industry / region / product / rep
# ---------------------------------------------------------------------------

def _segment_query(group_col):
    """Generic SQL template for segmentation: count, value, win rate per group."""
    return f"""
        SELECT
            {group_col} AS segment,
            COUNT(*) AS total_deals,
            SUM(CASE WHEN is_won = 1 THEN 1 ELSE 0 END) AS won_deals,
            SUM(CASE WHEN current_stage = 'Closed Lost' THEN 1 ELSE 0 END) AS lost_deals,
            SUM(CASE WHEN is_won = 1 THEN amount_eur ELSE 0 END) AS won_value_eur,
            SUM(CASE WHEN is_closed = 0 THEN amount_eur ELSE 0 END) AS active_pipeline_eur,
            ROUND(AVG(CASE WHEN is_won = 1 THEN amount_eur END), 0) AS avg_won_deal_size_eur
        FROM opportunities
        GROUP BY {group_col}
    """


def _add_win_rate(df):
    """Adds a win_rate column: won / (won + lost)."""
    closed = df["won_deals"] + df["lost_deals"]
    df["win_rate"] = (df["won_deals"] / closed.where(closed > 0, 1)).round(3)
    df.loc[closed == 0, "win_rate"] = 0.0
    return df


def pipeline_by_industry(db_path=None):
    df = pd.read_sql(_segment_query("industry"), get_engine(db_path))
    return _add_win_rate(df).sort_values("won_value_eur", ascending=False).reset_index(drop=True)


def pipeline_by_region(db_path=None):
    df = pd.read_sql(_segment_query("region"), get_engine(db_path))
    return _add_win_rate(df).sort_values("won_value_eur", ascending=False).reset_index(drop=True)


def pipeline_by_product(db_path=None):
    df = pd.read_sql(_segment_query("product_name"), get_engine(db_path))
    return _add_win_rate(df).sort_values("won_value_eur", ascending=False).reset_index(drop=True)


def pipeline_by_rep(db_path=None):
    """Per-rep performance, including their region for context."""
    engine = get_engine(db_path)
    query = """
        SELECT
            o.rep_name AS segment,
            r.region AS rep_region,
            COUNT(*) AS total_deals,
            SUM(CASE WHEN o.is_won = 1 THEN 1 ELSE 0 END) AS won_deals,
            SUM(CASE WHEN o.current_stage = 'Closed Lost' THEN 1 ELSE 0 END) AS lost_deals,
            SUM(CASE WHEN o.is_won = 1 THEN o.amount_eur ELSE 0 END) AS won_value_eur,
            SUM(CASE WHEN o.is_closed = 0 THEN o.amount_eur ELSE 0 END) AS active_pipeline_eur,
            ROUND(AVG(CASE WHEN o.is_won = 1 THEN o.amount_eur END), 0) AS avg_won_deal_size_eur
        FROM opportunities o
        JOIN sales_reps r ON o.rep_id = r.rep_id
        GROUP BY o.rep_name, r.region
    """
    df = pd.read_sql(query, engine)
    return _add_win_rate(df).sort_values("won_value_eur", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Time-series trends
# ---------------------------------------------------------------------------

def monthly_pipeline_trend(db_path=None):
    """
    For each month, compute:
        - won_value: total closed-won EUR that month
        - won_count: number of deals closed that month
        - new_pipeline: EUR added to the pipeline (deals created that month)
    """
    engine = get_engine(db_path)
    query = """
        SELECT
            strftime('%Y-%m', close_date) AS month,
            SUM(CASE WHEN is_won = 1 THEN amount_eur ELSE 0 END) AS won_value_eur,
            SUM(CASE WHEN is_won = 1 THEN 1 ELSE 0 END) AS won_count
        FROM opportunities
        WHERE close_date IS NOT NULL
        GROUP BY month
        ORDER BY month
    """
    won_df = pd.read_sql(query, engine)
    new_query = """
        SELECT
            strftime('%Y-%m', created_date) AS month,
            SUM(amount_eur) AS new_pipeline_eur,
            COUNT(*) AS new_deal_count
        FROM opportunities
        GROUP BY month
        ORDER BY month
    """
    new_df = pd.read_sql(new_query, get_engine(db_path))
    return won_df.merge(new_df, on="month", how="outer").fillna(0).sort_values("month").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Forecasting + risk
# ---------------------------------------------------------------------------

def weighted_pipeline(db_path=None):
    """
    Weighted pipeline value = sum(amount × stage_probability) for active deals.
    This is the standard "expected revenue" forecast metric.
    """
    engine = get_engine(db_path)
    df = pd.read_sql(
        "SELECT current_stage, amount_eur FROM opportunities WHERE is_closed = 0",
        engine,
    )
    df["weight"] = df["current_stage"].map(STAGE_WIN_PROBABILITY)
    df["weighted_eur"] = df["amount_eur"] * df["weight"]
    summary = (
        df.groupby("current_stage")
        .agg(
            deal_count=("amount_eur", "count"),
            total_value_eur=("amount_eur", "sum"),
            weight=("weight", "first"),
            weighted_value_eur=("weighted_eur", "sum"),
        )
        .reset_index()
        .rename(columns={"current_stage": "stage"})
    )
    stage_order = ["Prospecting", "Qualified", "Proposal", "Negotiation"]
    summary["stage"] = pd.Categorical(summary["stage"], categories=stage_order, ordered=True)
    return summary.sort_values("stage").reset_index(drop=True)


def top_at_risk_deals(db_path=None, n=10):
    """
    Identifies active deals that have been open longer than the typical deal at
    their current stage — flagging stuck opportunities that need rep attention.

    'At risk' = active deal with days_open > 1.5x the median for its stage.
    """
    engine = get_engine(db_path)
    df = pd.read_sql(
        "SELECT * FROM opportunities WHERE is_closed = 0",
        engine,
    )
    if df.empty:
        return df
    median_by_stage = df.groupby("current_stage")["days_open"].median().to_dict()
    df["stage_median_days"] = df["current_stage"].map(median_by_stage)
    df["risk_ratio"] = df["days_open"] / df["stage_median_days"].replace(0, 1)
    at_risk = df[df["risk_ratio"] >= 1.5].sort_values("risk_ratio", ascending=False)
    cols = [
        "opportunity_id", "account_id", "rep_name", "product_name", "industry", "region",
        "current_stage", "amount_eur", "days_open", "risk_ratio",
    ]
    return at_risk[cols].head(n).reset_index(drop=True)


def loss_reason_breakdown(db_path=None):
    """Why deals are lost. Useful for the memo's 'reduce churn' insights."""
    engine = get_engine(db_path)
    query = """
        SELECT
            loss_reason,
            COUNT(*) AS lost_count,
            ROUND(SUM(amount_eur), 0) AS lost_value_eur,
            ROUND(AVG(amount_eur), 0) AS avg_lost_deal_size_eur
        FROM opportunities
        WHERE current_stage = 'Closed Lost' AND loss_reason IS NOT NULL
        GROUP BY loss_reason
        ORDER BY lost_count DESC
    """
    return pd.read_sql(query, get_engine(db_path))