"""Quick smoke test for src/analytics.py — run all functions, eyeball outputs."""
import sys
from pathlib import Path

# Make the src/ folder importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analytics


def banner(title):
    print(f"\n{'=' * 70}\n {title}\n{'=' * 70}")


banner("PIPELINE SUMMARY")
print(analytics.pipeline_summary().T)

banner("BY STAGE")
print(analytics.pipeline_by_stage())

banner("STAGE CONVERSION RATES")
print(analytics.stage_conversion_rates())

banner("AVG DAYS IN STAGE")
print(analytics.avg_days_in_stage())

banner("BY INDUSTRY")
print(analytics.pipeline_by_industry())

banner("BY REGION")
print(analytics.pipeline_by_region())

banner("BY PRODUCT")
print(analytics.pipeline_by_product())

banner("BY REP (top 5)")
print(analytics.pipeline_by_rep().head(5))

banner("MONTHLY TREND (head + tail)")
trend = analytics.monthly_pipeline_trend()
print(trend.head(3))
print("...")
print(trend.tail(3))

banner("WEIGHTED PIPELINE")
print(analytics.weighted_pipeline())
total_weighted = analytics.weighted_pipeline()["weighted_value_eur"].sum()
print(f"\nTotal weighted pipeline forecast: EUR {total_weighted/1e6:.2f}M")

banner("TOP AT-RISK DEALS")
print(analytics.top_at_risk_deals(n=5))

banner("LOSS REASONS")
print(analytics.loss_reason_breakdown())

print("\nAll analytics functions executed without errors.")