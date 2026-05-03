"""
Pipeline Pulse — synthetic industrial-software sales pipeline data generator.

Produces 18 months (Nov 2024 – Apr 2026) of realistic B2B sales pipeline data
for a hypothetical industrial-software vendor. Output: SQLite database with
5 related tables (accounts, sales_reps, products, opportunities, stage_history).

Design principles encoded in this generator:
    - Industry-specific win rates and deal sizes
    - Product-specific deal-size distributions
    - Realistic stage progression with attrition at each step
    - Q4 close-rate spike (German enterprise budget cycle)
    - Regional effects (DACH home-turf advantage, APAC slower cycles)
    - Sales rep performance variance (top performers vs. average)
    - Realistic loss-reason categories
    - Time-in-stage scaling with deal size

Run from project root:
    python scripts/generate_pipeline_data.py
"""

import os
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Reproducibility — deterministic data so the dashboard tells the same story
# every time someone clones the repo.
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_US")
Faker.seed(SEED)

# ---------------------------------------------------------------------------
# Configuration — date range, scale, and reproducibility
# ---------------------------------------------------------------------------
START_DATE = datetime(2024, 11, 1)
END_DATE = datetime(2026, 4, 30)
N_ACCOUNTS = 180
N_OPPORTUNITIES = 600
N_SALES_REPS = 12

# Output paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROCESSED_DIR / "pipeline.db"

# ---------------------------------------------------------------------------
# Reference data — the static dimensions of the synthetic business
# ---------------------------------------------------------------------------

INDUSTRIES = {
    # Each industry has its own characteristic deal pattern
    "Automotive": {
        "weight": 0.30,           # 30% of accounts are automotive
        "win_rate_mod": 1.10,     # Slightly easier wins (mature market)
        "deal_size_mod": 1.20,    # Bigger deals on average
        "cycle_speed_mod": 0.90,  # Slightly faster sales cycle
    },
    "Aerospace": {
        "weight": 0.15,
        "win_rate_mod": 0.85,     # Harder wins (rigorous procurement)
        "deal_size_mod": 1.50,    # Very large deals
        "cycle_speed_mod": 1.40,  # Long cycles
    },
    "Pharma": {
        "weight": 0.20,
        "win_rate_mod": 0.95,
        "deal_size_mod": 1.30,
        "cycle_speed_mod": 1.20,  # Regulated industry, careful evaluation
    },
    "Energy": {
        "weight": 0.20,
        "win_rate_mod": 0.90,
        "deal_size_mod": 1.10,
        "cycle_speed_mod": 1.30,
    },
    "Electronics": {
        "weight": 0.15,
        "win_rate_mod": 1.05,
        "deal_size_mod": 0.85,    # Smaller deals, higher volume
        "cycle_speed_mod": 0.80,  # Fast-moving industry
    },
}

REGIONS = {
    "DACH": {
        "weight": 0.40,           # Home turf — most accounts here
        "win_rate_mod": 1.15,     # Best win rate
        "cycle_speed_mod": 0.95,
    },
    "EMEA-other": {
        "weight": 0.25,
        "win_rate_mod": 1.00,
        "cycle_speed_mod": 1.00,
    },
    "Americas": {
        "weight": 0.20,
        "win_rate_mod": 0.95,
        "cycle_speed_mod": 1.05,
    },
    "APAC": {
        "weight": 0.15,
        "win_rate_mod": 0.85,
        "cycle_speed_mod": 1.30,  # Longest sales cycles
    },
}

# Industrial-software product categories with realistic deal-size ranges.
# Generic names so we don't accidentally name a real Siemens/Mendix product.
PRODUCTS = {
    "PLM Suite": {
        "category": "Product Lifecycle Management",
        "min_acv": 100_000,
        "max_acv": 2_000_000,
        "median_acv": 350_000,
        "weight": 0.20,           # 20% of opportunities
    },
    "Low-Code Platform": {
        "category": "Application Development",
        "min_acv": 20_000,
        "max_acv": 300_000,
        "median_acv": 75_000,
        "weight": 0.30,           # Most popular product
    },
    "CAD Software": {
        "category": "Design & Engineering",
        "min_acv": 5_000,
        "max_acv": 150_000,
        "median_acv": 25_000,
        "weight": 0.20,
    },
    "MES": {
        "category": "Manufacturing Execution",
        "min_acv": 80_000,
        "max_acv": 800_000,
        "median_acv": 200_000,
        "weight": 0.15,
    },
    "Simulation Tools": {
        "category": "Engineering Simulation",
        "min_acv": 30_000,
        "max_acv": 500_000,
        "median_acv": 100_000,
        "weight": 0.15,
    },
}

# Sales pipeline stages, in order, with the typical conversion rate at each.
# E.g. only 60% of Prospecting deals make it to Qualified.
STAGES = ["Prospecting", "Qualified", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
ADVANCEMENT_PROBABILITY = {
    "Prospecting": 0.60,    # 40% die here
    "Qualified": 0.65,      # 35% die here
    "Proposal": 0.70,
    "Negotiation": 0.75,    # By here, deals are mostly going to close one way or another
}

# Average days a deal spends at each stage (will be scaled by deal size and region)
DAYS_AT_STAGE = {
    "Prospecting": 21,
    "Qualified": 28,
    "Proposal": 35,
    "Negotiation": 21,
}

# Realistic loss reasons with their relative frequencies
LOSS_REASONS = {
    "Lost to competitor": 0.30,
    "Budget cut / no funds": 0.25,
    "No decision / timing": 0.20,
    "Product fit": 0.15,
    "Internal champion left": 0.10,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def weighted_choice(d):
    """Pick a key from a dict where values are weights (must sum to 1.0)."""
    keys = list(d.keys())
    weights = [d[k]["weight"] if isinstance(d[k], dict) else d[k] for k in keys]
    return np.random.choice(keys, p=weights)


def sample_deal_size(product_name):
    """
    Sample a deal size from a log-normal-ish distribution clamped to product's range.
    Real B2B deal sizes are heavily right-skewed — most are small, a few are huge.
    """
    p = PRODUCTS[product_name]
    # Log-normal centered on the median
    log_median = np.log(p["median_acv"])
    sample = np.random.lognormal(mean=log_median, sigma=0.7)
    # Clamp to product's range
    return float(np.clip(sample, p["min_acv"], p["max_acv"]))


def quarter_of(date):
    """Return the calendar quarter as 'YYYY-Qn' for a given date."""
    return f"{date.year}-Q{(date.month - 1) // 3 + 1}"


def days_at_stage_for_deal(stage, deal_size, region):
    """
    Time at stage scales with deal size (bigger = slower) and region cycle speed.
    Adds Gaussian noise so the data isn't suspiciously uniform.
    """
    base = DAYS_AT_STAGE[stage]
    size_factor = 0.6 + 0.4 * np.log10(deal_size) / np.log10(1_000_000)  # bigger = slower
    region_factor = REGIONS[region]["cycle_speed_mod"]
    days = base * size_factor * region_factor
    days += np.random.normal(0, days * 0.2)  # ±20% noise
    return max(int(days), 3)  # never less than 3 days at a stage


def is_q4_boost(date):
    """
    Q4 (Oct-Dec) wins get a boost — German enterprise customers close before fiscal year-end.
    Adds realistic bunching of won deals in Nov-Dec.
    """
    return date.month in (10, 11, 12)


# ---------------------------------------------------------------------------
# Generators for each table
# ---------------------------------------------------------------------------

def generate_sales_reps():
    """Twelve sales reps distributed across regions, each with a performance multiplier."""
    rows = []
    region_pool = []
    for region, cfg in REGIONS.items():
        # Allocate reps proportional to region weight (rounded to integer)
        n = max(1, round(cfg["weight"] * N_SALES_REPS))
        region_pool.extend([region] * n)
    # Trim or pad to exactly N_SALES_REPS
    region_pool = region_pool[:N_SALES_REPS] + ["DACH"] * max(0, N_SALES_REPS - len(region_pool))

    for i, region in enumerate(region_pool[:N_SALES_REPS], start=1):
        # Performance multiplier: most reps are average, a few are stars
        perf = float(np.clip(np.random.normal(1.0, 0.18), 0.65, 1.45))
        rows.append({
            "rep_id": f"REP-{i:03d}",
            "rep_name": fake.name(),
            "region": region,
            "performance_multiplier": round(perf, 3),
        })
    return pd.DataFrame(rows)


def generate_products():
    """Five product rows."""
    rows = []
    for i, (name, cfg) in enumerate(PRODUCTS.items(), start=1):
        rows.append({
            "product_id": f"PROD-{i:02d}",
            "product_name": name,
            "category": cfg["category"],
            "median_acv_eur": cfg["median_acv"],
        })
    return pd.DataFrame(rows)


def generate_accounts():
    """180 manufacturing accounts."""
    rows = []
    for i in range(1, N_ACCOUNTS + 1):
        industry = weighted_choice(INDUSTRIES)
        region = weighted_choice(REGIONS)
        rows.append({
            "account_id": f"ACC-{i:04d}",
            "account_name": fake.company(),
            "industry": industry,
            "region": region,
            "country": _country_for_region(region),
            "employees": _employee_count_for_industry(industry),
        })
    return pd.DataFrame(rows)


def _country_for_region(region):
    pools = {
        "DACH": ["Germany", "Germany", "Germany", "Austria", "Switzerland"],
        "EMEA-other": ["France", "UK", "Italy", "Netherlands", "Spain", "Sweden"],
        "Americas": ["USA", "USA", "USA", "Canada", "Mexico", "Brazil"],
        "APAC": ["Japan", "South Korea", "China", "India", "Singapore", "Australia"],
    }
    return random.choice(pools[region])


def _employee_count_for_industry(industry):
    """Manufacturing companies skew larger; aerospace/auto are big employers."""
    base = {
        "Automotive": 8000,
        "Aerospace": 12000,
        "Pharma": 5000,
        "Energy": 6000,
        "Electronics": 3500,
    }[industry]
    return int(np.clip(np.random.lognormal(np.log(base), 0.9), 200, 200_000))


def generate_opportunities(accounts_df, reps_df, products_df):
    """
    Generate ~600 opportunities. Each opp:
      1. Picks an account, rep, product.
      2. Has a creation date uniformly across the 18-month window.
      3. Walks through the stage funnel — drops at each stage based on
         ADVANCEMENT_PROBABILITY, modified by industry/region/rep/Q4 effects.
      4. Records final stage (Closed Won, Closed Lost, or still in flight).
    Returns: opportunities_df, stage_history_df
    """
    opp_rows = []
    history_rows = []

    for i in range(1, N_OPPORTUNITIES + 1):
        # ----- Pick the participants -----
        account = accounts_df.sample(1).iloc[0]
        product_name = weighted_choice(PRODUCTS)
        product = products_df[products_df["product_name"] == product_name].iloc[0]

        # Sales rep is preferentially in the same region as the account, but not always
        same_region_reps = reps_df[reps_df["region"] == account["region"]]
        if len(same_region_reps) > 0 and random.random() < 0.75:
            rep = same_region_reps.sample(1).iloc[0]
        else:
            rep = reps_df.sample(1).iloc[0]

        # ----- Initial deal attributes -----
        created_date = START_DATE + timedelta(
            days=random.randint(0, (END_DATE - START_DATE).days)
        )
        deal_size = sample_deal_size(product_name)

        # ----- Walk the funnel -----
        current_stage = "Prospecting"
        current_date = created_date
        stages_visited = [(current_stage, current_date)]

        # The combined modifier that affects whether this deal advances
        win_mod = (
            INDUSTRIES[account["industry"]]["win_rate_mod"]
            * REGIONS[account["region"]]["win_rate_mod"]
            * rep["performance_multiplier"]
        )

        while current_stage in ADVANCEMENT_PROBABILITY:
            # Move forward in time
            days_here = days_at_stage_for_deal(current_stage, deal_size, account["region"])
            current_date += timedelta(days=days_here)

            # If we've run past the data window, opportunity is "still in flight"
            if current_date > END_DATE:
                current_date = END_DATE
                break

            # Decide: advance or die?
            advance_prob = ADVANCEMENT_PROBABILITY[current_stage] * win_mod
            # Q4 boost on late-stage wins
            if current_stage == "Negotiation" and is_q4_boost(current_date):
                advance_prob *= 1.15
            advance_prob = float(np.clip(advance_prob, 0.05, 0.95))

            if random.random() < advance_prob:
                # Advance to next stage
                next_stage_idx = STAGES.index(current_stage) + 1
                current_stage = STAGES[next_stage_idx]
            else:
                # Deal dies — Closed Lost
                current_stage = "Closed Lost"

            stages_visited.append((current_stage, current_date))

            if current_stage in ("Closed Won", "Closed Lost"):
                break

        # ----- Record the opportunity -----
        is_closed = current_stage in ("Closed Won", "Closed Lost")
        is_won = current_stage == "Closed Won"
        loss_reason = (
            np.random.choice(list(LOSS_REASONS.keys()), p=list(LOSS_REASONS.values()))
            if current_stage == "Closed Lost"
            else None
        )
        days_open = (current_date - created_date).days

        opp_rows.append({
            "opportunity_id": f"OPP-{i:05d}",
            "account_id": account["account_id"],
            "rep_id": rep["rep_id"],
            "product_id": product["product_id"],
            "product_name": product_name,
            "industry": account["industry"],
            "region": account["region"],
            "country": account["country"],
            "rep_name": rep["rep_name"],
            "amount_eur": round(deal_size, 2),
            "current_stage": current_stage,
            "is_closed": is_closed,
            "is_won": is_won,
            "created_date": created_date.date().isoformat(),
            "close_date": current_date.date().isoformat() if is_closed else None,
            "loss_reason": loss_reason,
            "days_open": days_open,
            "create_quarter": quarter_of(created_date),
            "close_quarter": quarter_of(current_date) if is_closed else None,
        })

        # ----- Record stage history -----
        for j, (stage, date) in enumerate(stages_visited):
            history_rows.append({
                "history_id": f"HIST-{i:05d}-{j:02d}",
                "opportunity_id": f"OPP-{i:05d}",
                "stage": stage,
                "entered_at": date.date().isoformat(),
                "stage_order": j,
            })

    return pd.DataFrame(opp_rows), pd.DataFrame(history_rows)


# ---------------------------------------------------------------------------
# Main: generate everything, save to SQLite + parquet
# ---------------------------------------------------------------------------

def main():
    print("Pipeline Pulse data generator")
    print(f"  Date range:   {START_DATE.date()} to {END_DATE.date()}")
    print(f"  Accounts:     {N_ACCOUNTS}")
    print(f"  Sales reps:   {N_SALES_REPS}")
    print(f"  Products:     {len(PRODUCTS)}")
    print(f"  Opportunities target: {N_OPPORTUNITIES}")
    print()

    # Ensure output directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Generate all tables in dependency order
    print("Generating sales reps...")
    reps_df = generate_sales_reps()

    print("Generating products...")
    products_df = generate_products()

    print("Generating accounts...")
    accounts_df = generate_accounts()

    print("Generating opportunities + stage history (this may take ~10 seconds)...")
    opps_df, history_df = generate_opportunities(accounts_df, reps_df, products_df)

    # ----- Save raw CSVs (for Power BI later) -----
    print("\nWriting raw CSVs to data/raw/...")
    accounts_df.to_csv(RAW_DIR / "accounts.csv", index=False)
    reps_df.to_csv(RAW_DIR / "sales_reps.csv", index=False)
    products_df.to_csv(RAW_DIR / "products.csv", index=False)
    opps_df.to_csv(RAW_DIR / "opportunities.csv", index=False)
    history_df.to_csv(RAW_DIR / "stage_history.csv", index=False)

    # ----- Save SQLite DB (for Streamlit + analytics) -----
    print(f"Writing SQLite database to {DB_PATH}...")
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    accounts_df.to_sql("accounts", conn, index=False)
    reps_df.to_sql("sales_reps", conn, index=False)
    products_df.to_sql("products", conn, index=False)
    opps_df.to_sql("opportunities", conn, index=False)
    history_df.to_sql("stage_history", conn, index=False)

    # Add useful indexes for the analytics queries we'll run
    cur = conn.cursor()
    cur.execute("CREATE INDEX idx_opp_account ON opportunities(account_id)")
    cur.execute("CREATE INDEX idx_opp_rep ON opportunities(rep_id)")
    cur.execute("CREATE INDEX idx_opp_product ON opportunities(product_id)")
    cur.execute("CREATE INDEX idx_opp_stage ON opportunities(current_stage)")
    cur.execute("CREATE INDEX idx_opp_close_quarter ON opportunities(close_quarter)")
    cur.execute("CREATE INDEX idx_history_opp ON stage_history(opportunity_id)")
    conn.commit()
    conn.close()

    # ----- Sanity-check summary -----
    print("\n--- Generation summary ---")
    print(f"Accounts:        {len(accounts_df):,}")
    print(f"Sales reps:      {len(reps_df):,}")
    print(f"Products:        {len(products_df):,}")
    print(f"Opportunities:   {len(opps_df):,}")
    print(f"Stage history:   {len(history_df):,}")
    print()
    won = (opps_df["current_stage"] == "Closed Won").sum()
    lost = (opps_df["current_stage"] == "Closed Lost").sum()
    open_ = (~opps_df["is_closed"]).sum()
    won_value = opps_df.loc[opps_df["is_won"], "amount_eur"].sum()
    pipeline_value = opps_df.loc[~opps_df["is_closed"], "amount_eur"].sum()
    win_rate = won / (won + lost) if (won + lost) > 0 else 0
    print(f"Closed Won:      {won:,}  (€{won_value/1e6:.1f}M total)")
    print(f"Closed Lost:     {lost:,}")
    print(f"Open pipeline:   {open_:,}  (€{pipeline_value/1e6:.1f}M total)")
    print(f"Win rate:        {win_rate:.1%}")
    print()
    print(f"By industry:")
    print(opps_df.groupby("industry")["amount_eur"].agg(["count", "mean", "sum"]).round(0))
    print()
    print("Done. SQLite DB at:", DB_PATH)


if __name__ == "__main__":
    main()