\# Pipeline Pulse



A self-contained, end-to-end sales pipeline analytics platform for industrial-software B2B operations. Synthetic Salesforce-style data, dual-frontend (Power BI + Streamlit), and a Groq-powered LLM chat agent — all wired to a single SQLite source of truth.



Built as a portfolio project to demonstrate practical data analytics, BI dashboarding, and agentic AI integration in a manufacturing-industry context.



!\[Streamlit Executive Summary](docs/screenshots/streamlit\_01\_executive.png)



\---



\## At a glance



\- \*\*18 months\*\* of synthetic pipeline data: \~800 opportunities, 180 manufacturing accounts, 5 industries, 4 regions, 5 industrial-software product lines

\- \*\*Realistic dynamics\*\*: log-normal deal-size distribution, region-specific win rates (DACH > APAC by 34 percentage points), Q4 close-rate boost, sales-rep performance variance, 5-category loss-reason taxonomy

\- \*\*13-function analytics layer\*\* in pure pandas/SQL — single source of truth shared by every frontend

\- \*\*Power BI dashboard\*\*: 4 pages (Executive Summary, Pipeline Health, Segment Analysis, Forecast \& Risk) with DAX measures, slicers, and conditional formatting

\- \*\*Streamlit dashboard\*\*: same 4 pages mirrored as a deployable Python app, with Plotly visualizations and a custom dark theme

\- \*\*Groq-powered chat agent\*\*: 5th Streamlit page with 7 LLM tools — natural-language → tool-calling → analytics → synthesized answer

\- \*\*Public deployment\*\*: live Streamlit Cloud URL (see below)



This project demonstrates the typical scope of an industrial-IT sales-analytics role:



| Analytics task | Where it's covered |

|---|---|

| Executive KPI dashboard | Executive Summary page (both frontends) |

| Pipeline funnel \& stage health | Pipeline Health page, weighted forecast measure |

| Segment analysis (industry / region / product / rep) | Segment Analysis page |

| Forecasting \& risk monitoring | Forecast \& Risk page, at-risk-deal detection logic |

| Natural-language data interface (GenAI / agentic) | Ask the Data page (Groq chat agent) |

| Reproducible analytics layer | `src/analytics.py` (13 functions) |



\---



\## Quick start



\### Prerequisites

\- Python 3.11

\- A Groq API key (free tier is sufficient) — set as `GROQ\_API\_KEY` in `.env`

\- (Optional) Power BI Desktop, if you want to inspect/modify the .pbix file



\### 1. Set up the environment



```bash

conda create -n pipeline-pulse python=3.11 -y

conda activate pipeline-pulse

pip install streamlit==1.56.0 pandas==2.2.3 numpy==1.26.4 plotly==5.24.1 \\

&#x20;           sqlalchemy==2.0.36 groq==0.13.0 python-dotenv==1.0.1 \\

&#x20;           fastparquet==2024.11.0 faker==33.1.0

```



\### 2. Configure your Groq key



Create a `.env` file in the project root:

GROQ\_API\_KEY=gsk\_your\_key\_here



\### 3. Generate the synthetic data



```bash

python scripts/generate\_pipeline\_data.py

```



Outputs:

\- `data/raw/\*.csv` — 5 CSV files (one per table)

\- `data/processed/pipeline.db` — SQLite database with foreign-key indexes



Generation is deterministic (`random.seed(42)`), so anyone cloning the repo gets the same data.



\### 4. Run the Streamlit dashboard



```bash

streamlit run ui/app.py

```



Opens at `http://localhost:8501`.



\### 5. (Optional) Open the Power BI report



Open `powerbi/pipeline-pulse.pbix` in Power BI Desktop. You'll need a SQLite ODBC driver configured to point at `data/processed/pipeline.db` — see notes in the file for setup details.



\---



\## Architecture

┌─────────────────────────────────────────────────────────────┐

│  generate\_pipeline\_data.py (Wiener-process simulator)       │

└───────────────────────────┬─────────────────────────────────┘

▼

┌─────────────────────┐

│  pipeline.db        │   ← single source of truth

│  (SQLite)            │

│  5 tables, FK indexed│

└──────────┬──────────┘

│

┌──────────────┼──────────────┐

│              │              │

┌──────────▼─┐  ┌─────────▼────────┐  ┌──▼────────────┐

│  Power BI   │  │  src/analytics.py │  │  src/agent.py │

│  .pbix      │  │  (13 functions)   │  │  (Groq agent) │

│  4 pages    │  │                   │  │  7 tools      │

└─────────────┘  └─────────┬─────────┘  └──────┬────────┘

│                   │

└───────┬───────────┘

▼

┌──────────────────────┐

│   Streamlit app      │

│   ui/app.py          │

│   5 pages            │

│   (4 dashboard +     │

│    1 chat agent)     │

└──────────────────────┘



The shared analytics layer is the design choice that holds everything together. The Power BI dashboard, the Streamlit dashboard, and the Groq agent all compute the same KPIs the same way — no version drift, no inconsistent numbers.



\---



\## The data model



\### Why synthetic data, and why this much care?



Most "I built a BI dashboard" portfolio projects use one of three things: a built-in Power BI sample, a Kaggle dataset, or `faker.fake\_business()` random rows. None of them have \*patterns\* a real analyst could extract insight from.



This generator produces data with deliberate, defensible structure. The full design is in `scripts/generate\_pipeline\_data.py`.



\### Key design choices



\#### Region-specific win-rate modifiers



```python

"DACH":       win\_rate\_mod = 1.15,   # home-turf advantage

"EMEA-other": win\_rate\_mod = 1.00,

"Americas":   win\_rate\_mod = 0.95,

"APAC":       win\_rate\_mod = 0.85,

```



Result: DACH wins 38.9% of closed deals, APAC wins 4.9%. This isn't a bug — it's the data telling the analyst that regional GTM strategy isn't uniform, which is a real pattern in international software sales.



\#### Industry-specific deal dynamics



Aerospace deals are largest by sticker price (1.5× modifier) but slowest to close (1.4× cycle modifier) and hardest to win (0.85× win-rate modifier). Automotive deals are 1.2× larger, slightly faster, slightly easier. Electronics deals are smallest (0.85×) but fast-moving (0.80× cycle).



This produces realistic per-industry win rates: Automotive 36.7%, Aerospace 13.8%, with the analytical implications visible in any segment chart.



\#### Log-normal deal-size distribution



Real B2B deal sizes are heavily right-skewed: most deals are small, a few are huge. The generator samples from a log-normal distribution clamped to each product's plausible range:



```python

def sample\_deal\_size(product\_name):

&#x20;   log\_median = np.log(p\["median\_acv"])

&#x20;   sample = np.random.lognormal(mean=log\_median, sigma=0.7)

&#x20;   return float(np.clip(sample, p\["min\_acv"], p\["max\_acv"]))

```



Result: PLM Suite has a median won-deal size of €366K with a long tail toward €2M; CAD Software has a median of €34K with a tail to €150K. The Pareto pattern (top 10 deals = 25% of total revenue) is what falls out naturally.



\#### Realistic stage progression



Each deal walks the funnel one stage at a time. The probability of advancing to the next stage is base-rate × industry-mod × region-mod × rep-performance — with a Q4 boost for late-stage deals (German fiscal-year close effect).



```python

ADVANCEMENT\_PROBABILITY = {

&#x20;   "Prospecting": 0.60,    # 40% die here

&#x20;   "Qualified": 0.65,      # 35% die here

&#x20;   "Proposal": 0.70,

&#x20;   "Negotiation": 0.75,

}

```



Result: a funnel that narrows correctly (54 → 47 → 24 → 20 active deals) with monotonically improving conversion rates as deals progress (61.6% → 64.1% → 73.1% → 79.2%).



\---



\## The four dashboard pages



Both Power BI and Streamlit implement the same four pages, by design.



\### Page 1 — Executive Summary



Top-of-funnel KPIs and the monthly trend.



!\[Executive Summary](docs/screenshots/streamlit\_01\_executive.png)



\- 4 KPI cards: Active Pipeline (€29.3M), Won Revenue (€27.3M), Win Rate (27.9%), Avg Won Deal Size (€149K)

\- Monthly Won Revenue trend line — shows the Q4 2025 → Q1 2026 acceleration

\- Top 10 won deals — €1.06M largest, PLM Suite + Automotive dominate



\### Page 2 — Pipeline Health



What's currently moving, broken down by stage and region.



!\[Pipeline Health](docs/screenshots/streamlit\_02\_pipeline\_health.png)



\- Active deals count (145), raw pipeline (€29.3M), weighted forecast (€9.0M)

\- Funnel: Prospecting €11.9M → Qualified €8.1M → Proposal €5.0M → Negotiation €4.4M

\- Stage × region stacked bar showing DACH dominance in early stages

\- Stage-to-stage conversion rate table



\### Page 3 — Segment Analysis



Where revenue comes from — and where it doesn't.



!\[Segment Analysis](docs/screenshots/streamlit\_03\_segment\_analysis.png)



\- Won Revenue by Industry (Automotive €11.6M leads, Aerospace €1.2M trails)

\- Win Rate by Region (DACH 38.9% green → APAC 4.9% red)

\- Active Pipeline by Product (PLM Suite is 42% of active value)

\- Top 10 sales reps with rates and revenue



\### Page 4 — Forecast \& Risk



Weighted-pipeline forecast and stuck-deal monitoring.



!\[Forecast \& Risk](docs/screenshots/streamlit\_04\_forecast\_risk.png)



\- Weighted Forecast (€9.0M), Total Pipeline raw (€29.3M), At-Risk Value (€2.6M)

\- Raw vs Weighted bars by stage — gap closes as confidence rises

\- Lost Revenue by Reason donut: "Lost to competitor" + "Budget cut" = €46.6M

\- Top at-risk active deals — open ≥1.5× the median for their stage



\---



\## The chat agent



`ui/app.py` page 5 — natural-language interface backed by Groq's `llama-3.3-70b-versatile` with 7 analytics tools.



!\[Ask the Data](docs/screenshots/streamlit\_05\_ask\_the\_data.png)



\### Architecture

User question

│

▼

Groq llama-3.3-70b-versatile (with TOOLS)

│

├─→ Decides which tool to call

▼

Tool dispatcher → analytics function → DataFrame

│

▼

Result returned to LLM

│

▼

LLM synthesizes 2-4 sentence answer



\### Tools exposed



| Tool | Returns |

|---|---|

| `get\_summary` | Top-level KPIs |

| `get\_pipeline\_by\_segment` | Industry / region / product / rep breakdown |

| `get\_funnel\_stages` | Active pipeline by stage |

| `get\_top\_won\_deals` | Largest won deals |

| `get\_at\_risk\_deals` | Stuck active deals |

| `get\_loss\_reasons` | Why deals are lost |

| `get\_monthly\_trend` | Won revenue over time |



\### Example interactions



> \*\*Q:\*\* Which industry has the lowest win rate?

> \*\*A:\*\* The Aerospace industry has the lowest win rate at 13.8%. This is significantly lower than the overall pipeline average of 27.9%.

> \*Tool used: `get\_pipeline\_by\_segment(segment='industry')`\*



> \*\*Q:\*\* Show me the top 5 sales reps by won revenue.

> \*\*A:\*\* The top 5 sales reps by won revenue are Daniel Wagner (€4.9M), Allison Hill (€4.2M), Abigail Shaffer (€4.0M), Angie Henderson (€3.9M), and Cristian Santos (€3.1M). Insight: DACH region reps dominate the top 5 spots.

> \*Tool used: `get\_pipeline\_by\_segment(segment='rep')`\*



The agent shows its work — every response includes which tool was called and exposes the underlying DataFrame in an expander, so users can verify the answer.



\---



\## Repository structure

pipeline-pulse/

├── data/

│   ├── raw/                       # Generated CSVs (gitignored)

│   └── processed/                 # SQLite DB (gitignored)

├── docs/

│   └── screenshots/               # Power BI + Streamlit screenshots

├── powerbi/

│   └── pipeline-pulse.pbix        # Power BI Desktop file (gitignored — binary)

├── scripts/

│   ├── generate\_pipeline\_data.py  # Synthetic data generator

│   ├── test\_analytics.py          # Smoke test for analytics layer

│   └── test\_agent.py              # Standalone CLI test for the chat agent

├── src/

│   ├── init.py

│   ├── analytics.py               # 13 KPI functions (single source of truth)

│   └── agent.py                   # Groq-powered chat agent + tool definitions

├── ui/

│   └── app.py                     # Streamlit multi-page dashboard

├── .env                           # Groq API key (gitignored)

├── .gitignore

└── README.md



\---



\## Lessons learned



A few things that come up only when you actually ship a multi-frontend BI project, not when you read tutorials.



\### One source of truth or none



The temptation when adding a Streamlit dashboard alongside a Power BI dashboard is to write the analytics logic twice — DAX measures in Power BI, pandas/SQL in Streamlit. This guarantees they will eventually disagree, which is a credibility-killer for any BI project. The fix is to push the canonical definitions into a single layer (`src/analytics.py`) and let both frontends call it. Win rate is computed in exactly one place: `won / (won + lost)`. If that ever needs to change, it changes once.



\### LLM tool descriptions are the actual prompt



When building the chat agent, I initially wrote terse tool descriptions like \*"Get win rate by industry."\* The agent picked correctly maybe 60% of the time. After expanding the descriptions to explicitly enumerate the dimensions (\*"Use 'industry' for vertical analysis (Automotive, Aerospace, Pharma, Energy, Electronics)"\*), tool selection accuracy hit \~95%. The LLM is reading the tool descriptions as part of its decision context — invest there, not in the system prompt.



\### SQLite + ODBC into Power BI is fiddlier than it looks



The official Power BI SQLite connector goes through ODBC, which means installing `sqliteodbc\_w64.exe` and configuring a System DSN that points at the absolute path of the .db file. This breaks the moment the .pbix is opened on a different machine. Workarounds: ship the data as CSVs alongside, or document the DSN setup explicitly in the README.



\### Date columns from SQLite import as Text



Power BI's ODBC connector imports SQLite date columns as Text by default (because SQLite has no native date type — it stores them as ISO strings). The fix is one click: select the column → Column tools → Data type → Date. Forgetting this means line charts treat dates as categorical labels and sort them alphabetically. \*Apr 2025\* sorts before \*Jan 2026\* — chronologically wrong.



\### "Top N by measure" filter on Power BI is genuinely buggy



Tried setting up a Top 10 by Won Revenue filter on the rep table. The Filter type dropdown silently strips the Top N option in some versions of Power BI Desktop 2.153. Fastest workaround: skip the filter, sort the column descending, and just accept showing all 12 reps. With small dimension tables this is fine; with large ones, it's a real limitation.



\---



\## Future work



\- \*\*SAP/Salesforce schema mapping\*\* — show how the same analytics layer would consume real CRM data with a thin schema-translation module

\- \*\*Streaming refresh\*\* — push new opportunity events into the SQLite DB on a schedule so the dashboards reflect "live" pipeline movement

\- \*\*Forecast accuracy backtesting\*\* — replay the 18-month data forward to evaluate whether the weighted-pipeline forecast was a useful leading indicator of actual closes

\- \*\*Multi-tenant deployment\*\* — wrap the analytics layer with row-level filters so multiple sales orgs could share the same dashboard with isolated views



\---



\## Why this project



Built to demonstrate end-to-end sales-analytics competence — synthetic data design, dual-frontend BI implementation, agentic AI integration, and the engineering discipline to keep them all consistent through a single analytics layer.



The dashboards aren't pretending to match any specific company's sales operation; the \*patterns\* in the data are realistic enough that the visualizations and the chat agent both produce defensible insights, and the code structure is portable to a real-data scenario with minimal change.



— Allen Mathew Charly

\[github.com/allencharly04](https://github.com/allencharly04) · \[linkedin.com/in/allen-mathew-charly-0157b8216](https://www.linkedin.com/in/allen-mathew-charly-0157b8216)

