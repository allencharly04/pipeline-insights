"""
Pipeline Pulse — Streamlit dashboard.

Multi-page Streamlit app mirroring the Power BI report. Connects to the same
SQLite database via the shared analytics layer (src/analytics.py).

Run from project root:
    streamlit run ui/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analytics
from src import agent

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pipeline Pulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — global filters and navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Pipeline Pulse")
st.sidebar.caption("Industrial-software sales pipeline analytics")

page = st.sidebar.radio(
    "Page",
    [
        "Executive Summary",
        "Pipeline Health",
        "Segment Analysis",
        "Forecast & Risk",
        "Ask the Data",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Synthetic data simulating an 18-month industrial-software pipeline. "
    "Built as a portfolio project."
)


# ---------------------------------------------------------------------------
# Helpers — formatting
# ---------------------------------------------------------------------------
def fmt_eur_m(value):
    """Format a number as €X.XM."""
    if value is None or pd.isna(value):
        return "—"
    return f"€{value / 1e6:.1f}M"


def fmt_eur_k(value):
    """Format a number as €XK."""
    if value is None or pd.isna(value):
        return "—"
    return f"€{value / 1e3:.0f}K"


def fmt_pct(value):
    """Format a decimal as XX.X%."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.1f}%"


# ---------------------------------------------------------------------------
# Page 1: Executive Summary
# ---------------------------------------------------------------------------
def page_executive_summary():
    st.title("📈 Executive Summary")
    st.caption("High-level pipeline health snapshot — Q2 2026")

    summary = analytics.pipeline_summary().iloc[0]

    # Top KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Pipeline", fmt_eur_m(summary["active_pipeline_value"]))
    c2.metric("Won Revenue", fmt_eur_m(summary["won_value"]))
    c3.metric("Win Rate", fmt_pct(summary["win_rate"]))
    c4.metric("Avg Won Deal Size", fmt_eur_k(summary["avg_won_deal_size"]))

    st.divider()

    # Monthly trend
    st.subheader("Monthly Won Revenue Trend")
    trend = analytics.monthly_pipeline_trend()
    fig = px.line(
        trend,
        x="month",
        y="won_value_eur",
        markers=True,
        labels={"month": "Month", "won_value_eur": "Won Revenue (€)"},
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
    fig.update_traces(line=dict(color="#1f77b4", width=3))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Top deals
    st.subheader("Top 10 Won Deals")
    engine = analytics.get_engine()
    top_deals = pd.read_sql(
        """
        SELECT
            account_id, product_name, industry, region, amount_eur
        FROM opportunities
        WHERE is_won = 1
        ORDER BY amount_eur DESC
        LIMIT 10
        """,
        engine,
    )
    top_deals["amount_eur"] = top_deals["amount_eur"].apply(
        lambda x: f"€{x:,.0f}"
    )
    top_deals.columns = ["Account", "Product", "Industry", "Region", "Deal Size"]
    st.dataframe(top_deals, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page 2: Pipeline Health
# ---------------------------------------------------------------------------
def page_pipeline_health():
    st.title("🌡️ Pipeline Health")
    st.caption("Active deals — funnel stage breakdown and weighted forecast")

    by_stage = analytics.pipeline_by_stage()
    weighted = analytics.weighted_pipeline()

    active_count = by_stage[by_stage["stage"].isin(
        ["Prospecting", "Qualified", "Proposal", "Negotiation"]
    )]["deal_count"].sum()
    weighted_total = weighted["weighted_value_eur"].sum()
    raw_active = by_stage[by_stage["stage"].isin(
        ["Prospecting", "Qualified", "Proposal", "Negotiation"]
    )]["total_value_eur"].sum()

    # Top KPI row
    c1, c2, c3 = st.columns(3)
    c1.metric("Active Deals", f"{int(active_count):,}")
    c2.metric("Total Pipeline (raw)", fmt_eur_m(raw_active))
    c3.metric("Weighted Forecast", fmt_eur_m(weighted_total))

    st.divider()

    # Two-column layout: funnel + stacked bar
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Active Pipeline Funnel")
        active_stages = by_stage[
            by_stage["stage"].isin(
                ["Prospecting", "Qualified", "Proposal", "Negotiation"]
            )
        ]
        fig = go.Figure(
            go.Funnel(
                y=active_stages["stage"].tolist(),
                x=active_stages["total_value_eur"].tolist(),
                textinfo="value+percent initial",
                texttemplate="€%{value:,.0f}",
                marker={"color": "#1f77b4"},
            )
        )
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Active Deals by Stage and Region")
        engine = analytics.get_engine()
        stage_region = pd.read_sql(
            """
            SELECT current_stage, region, COUNT(*) AS deal_count
            FROM opportunities
            WHERE is_closed = 0
            GROUP BY current_stage, region
            """,
            engine,
        )
        # Force stage order
        stage_order = ["Prospecting", "Qualified", "Proposal", "Negotiation"]
        stage_region["current_stage"] = pd.Categorical(
            stage_region["current_stage"], categories=stage_order, ordered=True
        )
        stage_region = stage_region.sort_values("current_stage")

        fig = px.bar(
            stage_region,
            x="deal_count",
            y="current_stage",
            color="region",
            orientation="h",
            labels={"deal_count": "Active Deals", "current_stage": ""},
        )
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Stage conversion rates
    st.subheader("Stage-to-Stage Conversion Rates")
    conv = analytics.stage_conversion_rates()
    conv_display = conv.copy()
    conv_display["conversion_rate"] = conv_display["conversion_rate"].apply(
        lambda x: f"{x * 100:.1f}%"
    )
    conv_display.columns = [
        "Stage",
        "Entered (count)",
        "Advanced (count)",
        "Conversion Rate",
    ]
    st.dataframe(conv_display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page 3: Segment Analysis
# ---------------------------------------------------------------------------
def page_segment_analysis():
    st.title("🎯 Segment Analysis")
    st.caption("Where revenue comes from — and where it doesn't")

    by_industry = analytics.pipeline_by_industry()
    by_region = analytics.pipeline_by_region()
    by_product = analytics.pipeline_by_product()
    by_rep = analytics.pipeline_by_rep()

    # Industry + Region row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Won Revenue by Industry")
        fig = px.bar(
            by_industry.sort_values("won_value_eur", ascending=False),
            x="segment",
            y="won_value_eur",
            labels={"segment": "Industry", "won_value_eur": "Won Revenue (€)"},
            color="won_value_eur",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            height=380, margin=dict(l=20, r=20, t=20, b=20), showlegend=False
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Win Rate by Region")
        fig = px.bar(
            by_region.sort_values("win_rate", ascending=True),
            x="win_rate",
            y="segment",
            orientation="h",
            labels={"win_rate": "Win Rate", "segment": "Region"},
            color="win_rate",
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(
            height=380, margin=dict(l=20, r=20, t=20, b=20), showlegend=False
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Product mix + Top reps
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Active Pipeline by Product")
        fig = px.pie(
            by_product,
            names="segment",
            values="active_pipeline_eur",
            hole=0.4,
        )
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Top 10 Sales Reps")
        top_reps = by_rep.head(10).copy()
        display_reps = pd.DataFrame(
            {
                "Sales Rep": top_reps["segment"],
                "Region": top_reps["rep_region"],
                "Total Deals": top_reps["total_deals"],
                "Won Revenue": top_reps["won_value_eur"].apply(
                    lambda x: f"€{x:,.0f}"
                ),
                "Win Rate": top_reps["win_rate"].apply(lambda x: f"{x * 100:.1f}%"),
            }
        )
        st.dataframe(display_reps, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page 4: Forecast & Risk
# ---------------------------------------------------------------------------
def page_forecast_risk():
    st.title("⚠️ Forecast & Risk")
    st.caption("Weighted pipeline forecast and at-risk deal monitoring")

    weighted = analytics.weighted_pipeline()
    losses = analytics.loss_reason_breakdown()
    at_risk = analytics.top_at_risk_deals(n=15)

    weighted_total = weighted["weighted_value_eur"].sum()
    raw_total = weighted["total_value_eur"].sum()
    risk_value = at_risk["amount_eur"].sum() if not at_risk.empty else 0

    # KPI row
    c1, c2, c3 = st.columns(3)
    c1.metric("Weighted Forecast", fmt_eur_m(weighted_total))
    c2.metric("Total Pipeline (raw)", fmt_eur_m(raw_total))
    c3.metric("At-Risk Deal Value", fmt_eur_m(risk_value))

    st.divider()

    # Raw vs weighted pipeline
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Raw vs Weighted Pipeline by Stage")
        comparison = weighted.copy()
        fig = go.Figure(
            data=[
                go.Bar(
                    name="Raw Pipeline",
                    x=comparison["stage"].astype(str),
                    y=comparison["total_value_eur"],
                    marker_color="#aec7e8",
                ),
                go.Bar(
                    name="Weighted Forecast",
                    x=comparison["stage"].astype(str),
                    y=comparison["weighted_value_eur"],
                    marker_color="#1f77b4",
                ),
            ]
        )
        fig.update_layout(
            barmode="group",
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="€",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Lost Revenue by Reason")
        fig = px.pie(
            losses,
            names="loss_reason",
            values="lost_value_eur",
            hole=0.4,
        )
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # At-risk deals
    st.subheader("Top At-Risk Active Deals")
    st.caption(
        "Active deals open ≥1.5× longer than the median for their current stage"
    )
    if at_risk.empty:
        st.info("No at-risk deals.")
    else:
        display_risk = at_risk.copy()
        display_risk["amount_eur"] = display_risk["amount_eur"].apply(
            lambda x: f"€{x:,.0f}"
        )
        display_risk["risk_ratio"] = display_risk["risk_ratio"].apply(
            lambda x: f"{x:.2f}×"
        )
        display_risk.columns = [
            "Opportunity",
            "Account",
            "Sales Rep",
            "Product",
            "Industry",
            "Region",
            "Stage",
            "Amount",
            "Days Open",
            "Risk Ratio",
        ]
        st.dataframe(display_risk, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Page 5: Ask the Data — Groq-powered chat agent
# ---------------------------------------------------------------------------
def page_ask_the_data():
    st.title("💬 Ask the Data")
    st.caption(
        "Natural-language interface to the pipeline. Ask anything — "
        "the agent decides which analytics function to call."
    )

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Example questions to bootstrap the user
    with st.expander("💡 Example questions to try", expanded=False):
        st.markdown(
            """
            - What's our overall win rate?
            - Which industry has the lowest win rate?
            - Show me the top 5 sales reps by won revenue.
            - How many deals are stuck in negotiation right now?
            - Why are we losing deals?
            - What's the trend of monthly won revenue?
            - Which region is our biggest growth opportunity?
            - What are the top 10 at-risk deals?
            """
        )

    # Render existing chat history
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("tool_used"):
                st.caption(
                    f"🔧 Tool: `{entry['tool_used']}({entry['tool_args']})`"
                )
            if entry.get("tool_result_df") is not None:
                with st.expander("Show underlying data"):
                    st.dataframe(
                        entry["tool_result_df"],
                        use_container_width=True,
                        hide_index=True,
                    )

    # Chat input
    user_question = st.chat_input("Ask about the pipeline...")
    if user_question:
        # Echo user question
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )
        with st.chat_message("user"):
            st.markdown(user_question)

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                result = agent.run_agent(user_question)
            st.markdown(result["answer"])
            if result.get("tool_used"):
                st.caption(
                    f"🔧 Tool: `{result['tool_used']}({result['tool_args']})`"
                )
            if result.get("tool_result_df") is not None and len(result["tool_result_df"]) > 0:
                with st.expander("Show underlying data"):
                    st.dataframe(
                        result["tool_result_df"],
                        use_container_width=True,
                        hide_index=True,
                    )

        # Save to history
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "tool_used": result.get("tool_used"),
                "tool_args": result.get("tool_args"),
                "tool_result_df": result.get("tool_result_df"),
            }
        )

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()
# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "Executive Summary":
    page_executive_summary()
elif page == "Pipeline Health":
    page_pipeline_health()
elif page == "Segment Analysis":
    page_segment_analysis()
elif page == "Forecast & Risk":
    page_forecast_risk()
elif page == "Ask the Data":
    page_ask_the_data()