# streamlit_app.py — Vacation Bidding & Optimization (Prototype)
# Author: ChatGPT
# Description:
#   A lightweight Streamlit app that allocates vacation requests fairly using a linear
#   optimization model (PuLP). Employees submit ranked requests; the solver maximizes
#   total satisfaction while respecting daily quotas (overall and per role) and adds
#   an optional fairness penalty so one person can’t monopolize prime time.
#
#   Features:
#   - Upload CSV of requests, or start from sample data.
#   - Ranked preferences (1=top choice → higher weight) with customizable weights.
#   - Global daily quota (e.g., max N people off/day).
#   - Optional per-role daily quotas (e.g., max 2 Dispatchers off/day, 1 Duty Manager/day).
#   - Whole-request approval (no partial splits), typical for vacation bidding.
#   - Fairness control: target awarded days per person + penalty for deviations.
#   - Outputs approved requests, per-employee summary, and a daily calendar view.
#   - Export results to CSV.
#
#   CSV Request Schema (headers, case-insensitive ok):
#     employee, role, start_date, end_date, rank
#   Dates in ISO (YYYY-MM-DD). Rank 1..5.
#
#   Example row:
#     Alice, Duty Manager, 2025-07-15, 2025-07-21, 1
#
#   Requirements (requirements.txt):
#     streamlit
#     pandas
#     pulp
#
#   Run:
#     streamlit run streamlit_app.py

import io
import sys
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import streamlit as st

try:
    import pulp
except Exception:
    st.warning("PuLP not found. Install with: pip install pulp")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pulp"], check=False)
    import pulp

# -----------------------------
# Helpers
# -----------------------------

def daterange(start: pd.Timestamp, end: pd.Timestamp):
    # Inclusive date range
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


BLOCK_LENGTH_DAYS = 4

GROUP_QUOTAS = {
    "Ops Leadership": {
        "roles": {
            "Duty Manager",
            "Operations Lead",
        },
        "quota": 2,
    },
    "Ops Support": {
        "roles": {
            "Flight Support",
            "Flight Support Lead",
            "Flight Follower",
            "Travel Coordinator",
            "Operations Control Lead",
        },
        "quota": 2,
    },
}


def clean_requests(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    required = ["employee", "role", "start_date", "end_date", "rank"]
    # Normalize column names to lower
    df = df.rename(columns={cols.get(c, c): c for c in cols})
    missing = [c for c in required if c not in df.columns.str.lower()]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Standardize
    df2 = pd.DataFrame({
        "employee": df[[c for c in df.columns if c.lower()=="employee"][0]].astype(str).str.strip(),
        "role": df[[c for c in df.columns if c.lower()=="role"][0]].astype(str).str.strip(),
        "start_date": pd.to_datetime(df[[c for c in df.columns if c.lower()=="start_date"][0]]),
        "end_date": pd.to_datetime(df[[c for c in df.columns if c.lower()=="end_date"][0]]),
        "rank": pd.to_numeric(df[[c for c in df.columns if c.lower()=="rank"][0]], errors="coerce").astype("Int64"),
    })

    # sanity checks
    if (df2["end_date"] < df2["start_date"]).any():
        raise ValueError("Found end_date earlier than start_date in one or more rows.")
    if df2["rank"].isna().any():
        raise ValueError("Rank must be numeric (1..5).")

    # Add request_id and length in days
    df2 = df2.reset_index(drop=True)
    df2["request_id"] = df2.index.astype(str)
    df2["days"] = (df2["end_date"] - df2["start_date"]).dt.days + 1

    if (df2["days"] != BLOCK_LENGTH_DAYS).any():
        raise ValueError(
            f"All requests must be exactly {BLOCK_LENGTH_DAYS} consecutive days to align with the 4-on/4-off schedule."
        )
    return df2


def expand_requests_days(df_req: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_req.iterrows():
        for d in daterange(r.start_date.normalize(), r.end_date.normalize()):
            rows.append({
                "request_id": r.request_id,
                "employee": r.employee,
                "role": r.role,
                "date": d.normalize(),
            })
    return pd.DataFrame(rows)


# -----------------------------
# UI: Sidebar Controls
# -----------------------------

st.set_page_config(page_title="Vacation Solver (Prototype)", layout="wide")
st.title("Vacation Bidding & Optimization — Prototype")

st.markdown(
"""
This prototype allocates vacation **requests (whole blocks)** using an optimization model:
- Maximizes weighted satisfaction by **rank** (1 = top choice → higher weight).
- Enforces **daily quotas** overall and optionally by **role**.
- Requires **4-day blocks** to match the 4-on/4-off schedule.
- Applies built-in **role group caps** (Ops Leadership + Ops Support limited to 2 off per day).
- Adds **fairness** by penalizing deviation from a target awarded days per person.

**Tip:** Start with the sample, then upload your CSV. Tune weights/quotas/fairness in the sidebar and re-run.
"""
)

with st.sidebar:
    st.header("Settings")

    st.subheader("Rank Weights (higher = more value)")
    w1 = st.number_input("Rank 1 weight", min_value=1.0, value=5.0, step=0.5)
    w2 = st.number_input("Rank 2 weight", min_value=0.0, value=3.0, step=0.5)
    w3 = st.number_input("Rank 3 weight", min_value=0.0, value=2.0, step=0.5)
    w4 = st.number_input("Rank 4 weight", min_value=0.0, value=1.0, step=0.5)
    w5 = st.number_input("Rank 5 weight", min_value=0.0, value=0.5, step=0.5)
    rank_weights = {1: w1, 2: w2, 3: w3, 4: w4, 5: w5}

    st.subheader("Quotas")
    global_quota = st.number_input("Max people off per day (global)", min_value=1, value=3, step=1)

    st.caption("Optional per-role daily quotas. Leave blank for none.")
    st.caption(
        "Built-in groups: Ops Leadership (Duty Manager + Operations Lead) and Ops Support "
        "(Flight Support family) are each limited to 2 people off per day."
    )
    role_quota_text = st.text_area(
        "Role quotas (JSON dict, e.g. {\"Dispatcher\": 2, \"Duty Manager\": 1})",
        value=""
    )
    role_quotas: Dict[str, int] = {}
    if role_quota_text.strip():
        try:
            role_quotas = dict(pd.read_json(io.StringIO(role_quota_text), typ='series'))
        except Exception:
            st.warning("Could not parse role quotas JSON. Ignoring.")
            role_quotas = {}

    st.subheader("Fairness")
    fairness_target = st.number_input("Target awarded days per person", min_value=0.0, value=10.0, step=1.0)
    fairness_lambda = st.number_input("Fairness penalty (L1)", min_value=0.0, value=0.2, step=0.1,
                                      help="Higher = stronger push toward equal awarded days around the target.")

    st.subheader("Seniority Minimums")
    st.caption(
        "Optionally require a minimum number of awarded days for specific employees (multiples of 4)."
    )
    seniority_json = st.text_area(
        "Minimum days per employee (JSON, e.g. {\"Avery\": 8, \"Brook\": 4})",
        value="",
    )
    min_days_per_employee: Dict[str, int] = {}
    if seniority_json.strip():
        try:
            parsed = dict(pd.read_json(io.StringIO(seniority_json), typ="series"))
            for name, days in parsed.items():
                if pd.isna(days):
                    continue
                days_int = int(days)
                if days_int % BLOCK_LENGTH_DAYS != 0:
                    st.warning(
                        f"Minimum days for {name} must be a multiple of {BLOCK_LENGTH_DAYS}. Ignoring this entry."
                    )
                    continue
                min_days_per_employee[name] = days_int
        except Exception:
            st.warning("Could not parse minimum days JSON. Ignoring.")
            min_days_per_employee = {}

    st.subheader("Other")
    season_start = st.date_input("Season start (optional, for calendar view)")
    season_end = st.date_input("Season end (optional, for calendar view)")

# -----------------------------
# Sample Data / Upload
# -----------------------------

st.subheader("1) Load Requests")

st.caption(
    f"Vacation bids must be submitted in {BLOCK_LENGTH_DAYS}-day blocks to mirror the 4-on/4-off schedule."
)

sample = pd.DataFrame({
    "employee": [
        "Avery", "Avery",
        "Brook",
        "Casey",
        "Devon",
        "Emery",
        "Finn",
        "Gray",
    ],
    "role": [
        "Duty Manager", "Operations Lead",
        "Flight Support",
        "Flight Support Lead",
        "Flight Follower",
        "Travel Coordinator",
        "Operations Control Lead",
        "Flight Support",
    ],
    "start_date": [
        "2025-07-01", "2025-07-09",
        "2025-07-05",
        "2025-07-13",
        "2025-07-17",
        "2025-07-21",
        "2025-07-25",
        "2025-07-29",
    ],
    "end_date": [
        "2025-07-04", "2025-07-12",
        "2025-07-08",
        "2025-07-16",
        "2025-07-20",
        "2025-07-24",
        "2025-07-28",
        "2025-08-01",
    ],
    "rank": [1, 3, 1, 2, 1, 2, 3, 2],
})

col1, col2 = st.columns([1,1])
with col1:
    st.write("**Sample requests (editable):**")
    edited_sample = st.data_editor(sample, num_rows="dynamic")
with col2:
    uploaded = st.file_uploader("Upload CSV (employee, role, start_date, end_date, rank)", type=["csv"]) 
    use_uploaded = st.checkbox("Use uploaded file (if provided)")

if use_uploaded and uploaded is not None:
    df_req_in = pd.read_csv(uploaded)
else:
    df_req_in = edited_sample.copy()

try:
    df_req = clean_requests(df_req_in)
except Exception as e:
    st.error(f"Error in requests: {e}")
    st.stop()

st.success(f"Loaded {len(df_req)} requests from {df_req['employee'].nunique()} employees.")

# -----------------------------
# Build Optimization Model
# -----------------------------

st.subheader("2) Solve Allocation")

if st.button("Run Optimization"):
    with st.spinner("Solving..."):
        df_days = expand_requests_days(df_req)
        if df_days.empty:
            st.warning("No request days to allocate.")
            st.stop()

        # Index sets
        R = df_req["request_id"].tolist()
        E = df_req["employee"].unique().tolist()
        D = sorted(df_days["date"].unique())

        # Decision variables: x[r] = 1 if request r approved, else 0 (whole-block approval)
        x = pulp.LpVariable.dicts("x", R, lowBound=0, upBound=1, cat=pulp.LpBinary)

        # Awarded days per employee: y[e] = sum(days(r) * x[r] for r by e)
        y = pulp.LpVariable.dicts("y", E, lowBound=0, cat=pulp.LpContinuous)

        # Deviation from target: y[e] - target = dev_pos[e] - dev_neg[e]
        dev_pos = pulp.LpVariable.dicts("dev_pos", E, lowBound=0, cat=pulp.LpContinuous)
        dev_neg = pulp.LpVariable.dicts("dev_neg", E, lowBound=0, cat=pulp.LpContinuous)

        model = pulp.LpProblem("VacationAllocation", pulp.LpMaximize)

        # Map handy lookups
        req_by_id = df_req.set_index("request_id").to_dict(orient="index")

        # Rank-weighted value
        def weight_for_rank(r):
            rk = int(req_by_id[r]["rank"]) if pd.notna(req_by_id[r]["rank"]) else 5
            return float(rank_weights.get(rk, 0.0))

        # Objective: maximize rank-weighted approvals minus fairness penalty
        obj = []
        for r in R:
            w = weight_for_rank(r)
            obj.append(w * x[r])
        if fairness_lambda > 0:
            for e in E:
                obj.append(-fairness_lambda * (dev_pos[e] + dev_neg[e]))
        model += pulp.lpSum(obj)

        # Constraints
        # 1) Daily global quota
        for d in D:
            # sum over requests that include day d
            reqs_on_d = df_days.loc[df_days["date"] == d, "request_id"].unique().tolist()
            model += pulp.lpSum([x[r] for r in reqs_on_d]) <= global_quota, f"global_quota_{d.date()}"

        # 2) Per-role daily quotas (optional)
        if role_quotas:
            for d in D:
                rows_d = df_days[df_days["date"] == d]
                for role, cap in role_quotas.items():
                    reqs_role_d = rows_d.loc[rows_d["role"] == role, "request_id"].unique().tolist()
                    if reqs_role_d:
                        model += pulp.lpSum([x[r] for r in reqs_role_d]) <= int(cap), f"role_{role}_{d.date()}"

        # 2b) Fixed role group quotas
        for d in D:
            rows_d = df_days[df_days["date"] == d]
            for group_name, cfg in GROUP_QUOTAS.items():
                reqs_group_d = rows_d.loc[rows_d["role"].isin(cfg["roles"]), "request_id"].unique().tolist()
                if reqs_group_d:
                    model += (
                        pulp.lpSum([x[r] for r in reqs_group_d])
                        <= int(cfg["quota"])
                    ), f"group_{group_name}_{d.date()}"

        # 3) Define awarded days y[e]
        for e in E:
            requests_e = df_req[df_req["employee"] == e]
            model += y[e] == pulp.lpSum([int(row.days) * x[row.request_id] for _, row in requests_e.iterrows()])

        # 4) Deviation linearization: y[e] - target = dev_pos[e] - dev_neg[e]
        for e in E:
            model += y[e] - float(fairness_target) == dev_pos[e] - dev_neg[e]

        # 5) Minimum awarded days by seniority (optional)
        for e in E:
            if e in min_days_per_employee:
                model += y[e] >= float(min_days_per_employee[e]), f"min_days_{e}"

        # Solve
        solver = pulp.PULP_CBC_CMD(msg=False)
        result_status = model.solve(solver)

    st.write(f"Status: **{pulp.LpStatus[result_status]}**  |  Objective: **{pulp.value(model.objective):.2f}**")

    # Extract solution
    df_req["approved"] = df_req["request_id"].apply(lambda r: int(pulp.value(x[r]) > 0.5))
    df_req["value"] = df_req["rank"].apply(lambda rk: rank_weights.get(int(rk), 0.0))

    approved = df_req[df_req["approved"] == 1].copy()
    st.success(f"Approved {len(approved)} of {len(df_req)} requests.")

    # Per-employee summary
    sum_emp = approved.groupby(["employee", "role"]).agg(
        requests=("request_id", "count"),
        days=("days", "sum"),
        avg_rank=("rank", "mean"),
        total_value=("value", "sum")
    ).reset_index()

    all_people = df_req[["employee", "role"]].drop_duplicates()
    sum_emp = (
        all_people.merge(sum_emp, on=["employee", "role"], how="left")
        .fillna({"requests": 0, "days": 0, "avg_rank": 0, "total_value": 0})
    )
    sum_emp["requests"] = sum_emp["requests"].astype(int)
    sum_emp["days"] = sum_emp["days"].astype(int)
    sum_emp = sum_emp.sort_values(["days", "total_value"], ascending=[False, False])

    if min_days_per_employee:
        sum_emp["min_days_required"] = sum_emp["employee"].map(min_days_per_employee).fillna(0).astype(int)

    colA, colB = st.columns([1,1])
    with colA:
        st.write("**Approved Requests**")
        st.dataframe(approved[["employee", "role", "start_date", "end_date", "days", "rank"]]
                     .sort_values(["start_date", "employee"]))
    with colB:
        st.write("**Per-Employee Summary**")
        st.dataframe(sum_emp)

    # Calendar View
    st.subheader("3) Calendar View (Daily Totals)")

    if not season_start or not season_end:
        # Auto-span from min to max approved dates
        if not approved.empty:
            cal_start = approved["start_date"].min().date()
            cal_end = approved["end_date"].max().date()
        else:
            cal_start = df_req["start_date"].min().date()
            cal_end = df_req["end_date"].max().date()
    else:
        cal_start = pd.to_datetime(season_start).date()
        cal_end = pd.to_datetime(season_end).date()

    df_days_all = expand_requests_days(approved)
    # daily totals
    all_dates = pd.date_range(cal_start, cal_end, freq="D")
    day_counts = (
        df_days_all.groupby("date")["employee"].nunique()
        .reindex(all_dates, fill_value=0)
        .rename("people_off")
        .to_frame()
    )
    st.line_chart(day_counts)

    # Daily roster table
    st.write("**Daily roster (who is off):**")
    roster_rows = []
    if not df_days_all.empty:
        for d in all_dates:
            who = df_days_all[df_days_all["date"] == d][["employee", "role"]]
            names = [f"{r.employee} ({r.role})" for r in who.itertuples(index=False)]
            roster_rows.append({"date": d.date(), "count": len(names), "names": ", ".join(sorted(names))})
    roster = pd.DataFrame(roster_rows)
    st.dataframe(roster)

    # Exports
    st.subheader("4) Export Results")
    out_alloc = approved[["employee", "role", "start_date", "end_date", "days", "rank"]].copy()
    csv_alloc = out_alloc.to_csv(index=False).encode("utf-8")
    st.download_button("Download Approved Requests (CSV)", data=csv_alloc, file_name="approved_requests.csv")

    out_summary = sum_emp.copy()
    csv_sum = out_summary.to_csv(index=False).encode("utf-8")
    st.download_button("Download Per-Employee Summary (CSV)", data=csv_sum, file_name="employee_summary.csv")

else:
    st.info("Adjust settings, then click **Run Optimization** above to compute an allocation.")

# -----------------------------
# Template CSV download
# -----------------------------

st.subheader("Template & Docs")

schema = pd.DataFrame({
    "employee": ["Alice"],
    "role": ["Duty Manager"],
    "start_date": ["2025-07-15"],
    "end_date": ["2025-07-21"],
    "rank": [1],
})

buf = io.StringIO()
schema.to_csv(buf, index=False)
st.download_button("Download CSV Template", buf.getvalue().encode("utf-8"), file_name="vacation_requests_template.csv")

st.markdown(
    """
**Notes**
- This model approves **whole requests** (start→end). If you prefer partial approvals, we can switch to day-level decision variables.
- To strongly discourage long blocks, add a negative weight proportional to days, or cap `days` per request in preprocessing.
- For **blackout periods** (e.g., special events), we can inject day-level constraints that set the quota to 0 or a lower cap.
- For **seniority** weighting, add an employee-specific coefficient in the objective.
- To guarantee senior employees receive more time off, use the sidebar's *Seniority Minimums* to require minimum awarded days per person (in 4-day blocks).
- For **carry-over fairness** year‑to‑year, store prior awarded days and shift each employee’s target accordingly.
"""
)
