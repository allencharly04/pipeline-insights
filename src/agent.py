"""
Pipeline Pulse — Groq-powered chat agent.

Implements a natural-language interface to the pipeline analytics layer using
Groq's tool-calling API. The agent receives a user question, decides which
analytics function to invoke, runs it, and synthesizes a plain-English answer
alongside the raw data.

Tools exposed to the LLM:
    - get_summary               (top-level KPIs)
    - get_pipeline_by_segment   (industry / region / product / rep breakdown)
    - get_funnel_stages         (active pipeline by stage)
    - get_top_won_deals         (largest won deals)
    - get_at_risk_deals         (stuck active deals)
    - get_loss_reasons          (why deals are lost)
    - get_monthly_trend         (won revenue over time)
"""

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from src import analytics

# Load API key from .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GROQ_MODEL = "llama-3.3-70b-versatile"


def get_groq_client():
    """Build a Groq client using the key from .env."""
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------------------------------------------------------
# Tool definitions — schemas the LLM sees
# ---------------------------------------------------------------------------
# Each tool tells the LLM: what it's called, what it does, what params it takes.
# When the LLM picks a tool, we run the matching Python function (below) and
# return the result.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": (
                "Get the headline pipeline KPIs: total opportunities, active "
                "pipeline value (open deals), won revenue, win rate, average "
                "won deal size, and average days to close. Use this for "
                "questions about overall performance."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pipeline_by_segment",
            "description": (
                "Break down the pipeline by a chosen segment dimension. "
                "Returns total deals, won deals, lost deals, won value, "
                "active pipeline value, win rate, and average won deal size "
                "for each value of the segment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segment": {
                        "type": "string",
                        "enum": ["industry", "region", "product", "rep"],
                        "description": (
                            "Which dimension to break down by. Use 'industry' "
                            "for vertical analysis (Automotive, Aerospace, "
                            "Pharma, Energy, Electronics), 'region' for "
                            "geographic (DACH, EMEA-other, Americas, APAC), "
                            "'product' for product line (PLM Suite, "
                            "Low-Code Platform, CAD Software, MES, "
                            "Simulation Tools), 'rep' for sales rep "
                            "performance."
                        ),
                    }
                },
                "required": ["segment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_funnel_stages",
            "description": (
                "Get active pipeline broken down by stage (Prospecting, "
                "Qualified, Proposal, Negotiation), with deal count and "
                "total value at each stage. Use for questions about "
                "pipeline shape, active deals, or stage distribution."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_won_deals",
            "description": (
                "Get the largest won deals by amount. Useful for questions "
                "about biggest customers, top revenue contributors, or "
                "marquee wins."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of top deals to return (1-50, default 10).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_at_risk_deals",
            "description": (
                "Get active deals that have been open longer than expected "
                "for their stage (>= 1.5x the median for that stage). "
                "Returns deals at risk of stalling or being lost. Use for "
                "questions about stuck deals, slow-moving opportunities, "
                "or pipeline health concerns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of at-risk deals to return (1-30, default 10).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_loss_reasons",
            "description": (
                "Get a breakdown of why deals are lost (Lost to competitor, "
                "Budget cut, No decision, Product fit, Internal champion "
                "left), with count and total value lost per reason."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_monthly_trend",
            "description": (
                "Get monthly won revenue and new pipeline added across the "
                "data window. Useful for questions about trends, "
                "seasonality, or growth over time."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations — the actual Python that runs
# ---------------------------------------------------------------------------

def _run_tool(name, args):
    """Dispatch a tool call to the matching analytics function."""
    if name == "get_summary":
        return analytics.pipeline_summary()
    if name == "get_pipeline_by_segment":
        seg = args.get("segment", "industry")
        if seg == "industry":
            return analytics.pipeline_by_industry()
        if seg == "region":
            return analytics.pipeline_by_region()
        if seg == "product":
            return analytics.pipeline_by_product()
        if seg == "rep":
            return analytics.pipeline_by_rep().head(15)
        raise ValueError(f"Unknown segment: {seg}")
    if name == "get_funnel_stages":
        return analytics.pipeline_by_stage()
    if name == "get_top_won_deals":
        n = int(args.get("n", 10))
        engine = analytics.get_engine()
        return pd.read_sql(
            f"""
            SELECT account_id, product_name, industry, region, rep_name,
                   amount_eur, close_date
            FROM opportunities
            WHERE is_won = 1
            ORDER BY amount_eur DESC
            LIMIT {min(max(n, 1), 50)}
            """,
            engine,
        )
    if name == "get_at_risk_deals":
        n = int(args.get("n", 10))
        return analytics.top_at_risk_deals(n=min(max(n, 1), 30))
    if name == "get_loss_reasons":
        return analytics.loss_reason_breakdown()
    if name == "get_monthly_trend":
        return analytics.monthly_pipeline_trend()
    raise ValueError(f"Unknown tool: {name}")


def _df_to_llm_text(df, max_rows=20):
    """
    Convert a DataFrame to a compact text representation for the LLM.
    LLMs read tables better as markdown / plain text than JSON.
    """
    if df is None or len(df) == 0:
        return "(no data returned)"
    if len(df) > max_rows:
        df = df.head(max_rows)
        suffix = f"\n(showing first {max_rows} rows)"
    else:
        suffix = ""
    return df.to_string(index=False) + suffix


# ---------------------------------------------------------------------------
# System prompt — the agent's persona and rules
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Pipeline Pulse, a sales pipeline analytics assistant.

You answer questions about an industrial-software B2B sales pipeline. The data
covers ~800 opportunities across 18 months (Nov 2024 - Apr 2026), 5 industries
(Automotive, Aerospace, Pharma, Energy, Electronics), 4 regions (DACH,
EMEA-other, Americas, APAC), and 5 product lines (PLM Suite, Low-Code Platform,
CAD Software, MES, Simulation Tools).

When the user asks a question:
1. Pick the right tool to fetch relevant data.
2. Read the result.
3. Answer in 2-4 sentences max — concise, specific, with actual numbers.
4. Always include the most relevant statistic (e.g., percentages, EUR values,
   counts).
5. Round large EUR values to nearest €100K (e.g., "€2.4M" not "€2,438,221.43").
6. Round percentages to 1 decimal place.
7. If the data shows a striking pattern, mention it as a one-line insight.

DO NOT:
- Make up numbers not present in the data.
- Hedge excessively ("it depends", "approximately").
- Repeat the user's question back.
- Add disclaimers about data being synthetic.

Be direct, professional, and useful. Think of yourself as a sharp sales ops
analyst answering a busy executive."""


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent(user_question, history=None):
    """
    Run one round of the agent on a user question.
    Returns: dict with keys:
        - answer (str): the natural-language reply
        - tool_used (str | None): name of the tool that was called
        - tool_args (dict): args passed to the tool
        - tool_result_df (DataFrame | None): the raw data returned

    history: optional list of prior messages for multi-turn context.
    """
    client = get_groq_client()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_question})

    # First call — let the LLM decide whether to use a tool
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.2,
        max_tokens=800,
    )

    msg = response.choices[0].message

    # If no tool needed, return direct answer
    if not msg.tool_calls:
        return {
            "answer": msg.content,
            "tool_used": None,
            "tool_args": {},
            "tool_result_df": None,
        }

    # Tool was called — run it
    tool_call = msg.tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments or "{}")

    try:
        tool_result_df = _run_tool(tool_name, tool_args)
    except Exception as e:
        return {
            "answer": f"I tried to look that up but hit an error: {e}",
            "tool_used": tool_name,
            "tool_args": tool_args,
            "tool_result_df": None,
        }

    # Send the tool result back to the LLM for synthesis
    messages.append(
        {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": _df_to_llm_text(tool_result_df),
        }
    )

    final_response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=600,
    )

    return {
        "answer": final_response.choices[0].message.content,
        "tool_used": tool_name,
        "tool_args": tool_args,
        "tool_result_df": tool_result_df,
    }