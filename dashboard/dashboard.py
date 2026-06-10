# dashboard/dashboard.py
import json
import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from shared.utils import get_api_headers, get_api_url

# Set page config
st.set_page_config(page_title="Tunisian Fraud Detection - Command Center", page_icon="🚨", layout="wide")

# Title
st.title("🚨 Tunisian Fraud Detection - Operational Command Center")
st.markdown("---")

# Initialize session state
if "selected_transaction" not in st.session_state:
    st.session_state.selected_transaction = None
if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0

# Sidebar for navigation and filters
with st.sidebar:
    st.header("📊 Dashboard Controls")

    # Refresh interval selection
    refresh_interval = st.selectbox("Refresh Interval (seconds)", options=[5, 10, 30, 60], index=1)

    # Determine role from API (RBAC - no user selection)
    token = os.getenv("COMMAND_CENTER_API_TOKEN", "default_dev_token")
    headers = get_api_headers()

    # Try to get user info from API
    try:
        whoami_response = requests.get(get_api_url("auth/whoami/"), headers=headers)
        if whoami_response.status_code == 200:
            user_info = whoami_response.json()
            role = user_info.get("role", "Analyst")
        else:
            # Fallback to checking token type if whoami endpoint doesn't exist
            if "analyst" in token.lower():
                role = "Analyst"
            elif "admin" in token.lower() or "default_dev_token" in token:
                role = "Admin"
            else:
                role = "Analyst"  # Default to analyst
    except:
        # Fallback if API is not available
        if "analyst" in token.lower():
            role = "Analyst"
        elif "admin" in token.lower() or "default_dev_token" in token:
            role = "Admin"
        else:
            role = "Analyst"  # Default to analyst

    st.write(f"**Role:** {role}")

    user_id = st.text_input("User ID", value=os.getenv("DASHBOARD_USER_ID", ""))

    # Filter options
    st.subheader("Filters")
    branch_id_filter = None
    branches = []
    try:
        headers = get_api_headers()
        branch_url = get_api_url("branches/")
        branch_response = requests.get(branch_url, headers=headers)
        if branch_response.status_code == 200:
            branches = branch_response.json()
    except Exception:
        branches = []

    if role == "Admin":
        branch_options = ["All"] + branches
        selected_branch = st.selectbox("Branch", options=branch_options, index=0)
        if selected_branch != "All":
            branch_id_filter = selected_branch
    else:
        if branches:
            branch_id_filter = st.selectbox("Branch", options=branches, index=0)
        else:
            branch_id_filter = st.text_input("Branch", value="")
            if not branch_id_filter:
                branch_id_filter = None

    min_prob = st.slider("Minimum ML Probability (High Risk)", 0.0, 1.0, 0.85)
    show_sar = st.checkbox("Show SAR Reports", value=True)
    alert_type_filter = st.selectbox(
        "Alert Type", options=["All", "High Risk", "Random Sample", "Uncertainty Sample"], index=0
    )

    # Stats refresh button
    if st.button("🔄 Refresh Statistics"):
        st.session_state.refresh_counter += 1
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Review Queue (High Risk + Random Samples)")

    # Fetch alerts from API
    try:
        # Use the shared utility functions
        headers = get_api_headers()
        alert_type_param = None
        if alert_type_filter == "High Risk":
            alert_type_param = "high_risk"
        elif alert_type_filter == "Random Sample":
            alert_type_param = "random_sample"
        elif alert_type_filter == "Uncertainty Sample":
            alert_type_param = "uncertainty_sample"

        if alert_type_param:
            api_url = get_api_url(f"alerts/review-queue/?limit=50&alert_type={alert_type_param}")
        else:
            api_url = get_api_url("alerts/review-queue/?limit=50")
        if branch_id_filter:
            api_url = f"{api_url}&branch_id={branch_id_filter}"

        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            alerts = response.json()

            if alerts:
                # Convert to DataFrame for easier manipulation
                df = pd.DataFrame(alerts)

                if "alert_type" not in df.columns:
                    df["alert_type"] = "high_risk"
                df["alert_type"] = df["alert_type"].fillna("high_risk")

                # Filter by minimum probability for high-risk alerts only
                keep_mask = (df["alert_type"] != "high_risk") | (df["ml_probability"] >= min_prob)
                df_filtered = df[keep_mask]

                if not df_filtered.empty:
                    df_display = df_filtered.copy()
                    df_display["alert_type_display"] = (
                        df_display["alert_type"]
                        .map(
                            {
                                "high_risk": "High Risk",
                                "random_sample": "Random Sample",
                                "uncertainty_sample": "Uncertainty Sample",
                            }
                        )
                        .fillna("High Risk")
                    )

                    # Display alerts in a table
                    df_table = df_display[
                        [
                            "transaction_id",
                            "user_id",
                            "amount_tnd",
                            "governorate",
                            "payment_method",
                            "branch_id",
                            "ml_probability",
                            "alert_type_display",
                        ]
                    ].rename(columns={"alert_type_display": "alert_type"})
                    st.dataframe(
                        df_table.style.format({"amount_tnd": "{:.2f}", "ml_probability": "{:.3f}"}),
                        use_container_width=True,
                        height=400,
                    )

                    # Show transaction details when selected
                    if len(df_display) > 0:
                        selected_idx = st.selectbox(
                            "Select Transaction for Details",
                            options=range(len(df_display)),
                            format_func=lambda x: (
                                f"{df_display.iloc[x]['transaction_id']} - "
                                f"{df_display.iloc[x]['amount_tnd']:.2f} TND "
                                f"({df_display.iloc[x]['alert_type_display']})"
                            ),
                        )

                        if selected_idx is not None:
                            selected_row = df_display.iloc[selected_idx]
                            st.session_state.selected_transaction = selected_row.to_dict()
                else:
                    st.info("No alerts matching current filters.")
            else:
                st.info("No alerts available.")
        else:
            st.error(f"Failed to fetch alerts: {response.status_code}")
    except requests.exceptions.ConnectionError:
        api_url_display = os.getenv("COMMAND_CENTER_API_URL", "http://localhost:8001")
        st.error(f"Could not connect to the API. Please ensure the FastAPI server is running on {api_url_display}")
    except Exception as e:
        st.error(f"Error fetching alerts: {str(e)}")

with col2:
    st.subheader("📊 System Statistics")

    # Fetch stats from API
    try:
        # Use the shared utility functions
        headers = get_api_headers()
        api_url = get_api_url("stats")
        if branch_id_filter:
            api_url = f"{api_url}?branch_id={branch_id_filter}"

        stats_response = requests.get(api_url, headers=headers)
        if stats_response.status_code == 200:
            stats = stats_response.json()

            # Display key metrics
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Total Feedback", stats.get("total_feedback", 0))
            with col2_2:
                st.metric("Alert Precision", f"{stats.get('precision', 0):.3f}")

            random_sample_fraud_rate = stats.get("random_sample_fraud_rate")
            if random_sample_fraud_rate is not None:
                st.metric("Random Sample Fraud Rate", f"{random_sample_fraud_rate:.3f}")

            st.metric("High-Risk Alerts", stats.get("high_risk_alerts", 0))
            st.metric("Random Samples", stats.get("random_sample_alerts", 0))
            st.metric("Uncertainty Samples", stats.get("uncertainty_sample_alerts", 0))
            st.metric("Review Queue Total", stats.get("review_queue_total", 0))
            if stats.get("random_sample_rate") is not None:
                st.caption(f"Random sample rate: {stats.get('random_sample_rate'):.3f}")

            compliance_url = get_api_url("compliance/kpis/")
            if branch_id_filter:
                compliance_url = f"{compliance_url}?branch_id={branch_id_filter}"
            compliance_response = requests.get(compliance_url, headers=headers)
            if compliance_response.status_code == 200:
                compliance = compliance_response.json()
                st.subheader("Compliance KPIs")
                kpi_col1, kpi_col2 = st.columns(2)
                with kpi_col1:
                    st.metric("SARs Filed (30d)", compliance.get("sar_reports_generated", 0))
                    st.metric("Sanctions Hits (30d)", compliance.get("sanctions_hits", 0))
                with kpi_col2:
                    st.metric("Overdue SARs", compliance.get("overdue_sar_count", 0))
                    st.metric("False Positive Rate", f"{compliance.get('false_positive_rate', 0):.2f}%")
                st.metric("SAR On-Time Rate", f"{compliance.get('sar_on_time_percent', 0):.2f}%")

                pkyc_by_reason = compliance.get("pkyc_triggers_by_reason", {})
                if pkyc_by_reason:
                    pkyc_df = pd.DataFrame(
                        list(pkyc_by_reason.items()),
                        columns=["Trigger Reason", "Count"],
                    )
                    st.bar_chart(pkyc_df.set_index("Trigger Reason"))

                accounts_by_tier = compliance.get("high_risk_accounts_by_tier", {})
                if accounts_by_tier:
                    tier_df = pd.DataFrame(
                        list(accounts_by_tier.items()),
                        columns=["Risk Tier", "Accounts"],
                    )
                    st.dataframe(tier_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Compliance KPIs unavailable.")

            # Feedback breakdown chart
            if stats.get("feedback_breakdown"):
                breakdown = stats["feedback_breakdown"]
                breakdown_df = pd.DataFrame(list(breakdown.items()), columns=["Label", "Count"])

                fig = px.pie(breakdown_df, values="Count", names="Label", title="Feedback Distribution")
                st.plotly_chart(fig, use_container_width=True)

            if role == "Admin":
                st.subheader("CTAF Export")
                export_url = get_api_url("alerts/ctaf-export")
                if branch_id_filter:
                    export_url = f"{export_url}?branch_id={branch_id_filter}"
                if st.button("Download CTAF Report"):
                    export_response = requests.get(export_url, headers=headers)
                    if export_response.status_code == 200:
                        export_payload = export_response.json()
                        st.download_button(
                            "Download JSON",
                            data=json.dumps(export_payload, indent=2),
                            file_name="ctaf_report.json",
                            mime="application/json",
                        )
                    else:
                        st.error("Failed to export CTAF report")

            perf_url = get_api_url("monitoring/model-performance/")
            if branch_id_filter:
                perf_url = f"{perf_url}?branch_id={branch_id_filter}"
            perf_response = requests.get(perf_url, headers=headers)
            if perf_response.status_code == 200:
                perf = perf_response.json()
                st.subheader("Model Performance (Sampling-Aware)")
                st.metric("Estimated Recall", f"{perf.get('recall', 0):.3f}")
                st.metric("Reviewed Recall", f"{perf.get('reviewed_recall', 0):.3f}")
                st.metric("Estimated False Negatives", perf.get("estimated_false_negatives", 0))
        else:
            st.error(f"Failed to fetch stats: {stats_response.status_code}")
    except requests.exceptions.ConnectionError:
        api_url_display = os.getenv("COMMAND_CENTER_API_URL", "http://localhost:8001")
        st.error(f"Could not connect to the API for statistics on {api_url_display}.")
    except Exception as e:
        st.error(f"Error fetching stats: {str(e)}")

# Detailed view for selected transaction
if st.session_state.selected_transaction:
    st.markdown("---")
    st.subheader("🔍 Transaction Details")

    trans = st.session_state.selected_transaction
    alert_type = trans.get("alert_type", "high_risk")
    if alert_type == "random_sample":
        alert_type_display = "Random Sample"
    elif alert_type == "uncertainty_sample":
        alert_type_display = "Uncertainty Sample"
    else:
        alert_type_display = "High Risk"

    if alert_type == "random_sample":
        st.info("Random sample review. Use 'False Positive' for non-fraud cases.")
    if alert_type == "uncertainty_sample":
        st.info("Uncertainty sample review. Provide analyst label to refine the model.")

    # Display transaction details
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Transaction ID:** {trans['transaction_id']}")
        st.write(f"**User ID:** {trans['user_id']}")
        st.write(f"**Amount:** {trans['amount_tnd']:.2f} TND")
        st.write(f"**Governorate:** {trans['governorate']}")
        st.write(f"**Payment Method:** {trans['payment_method']}")
        st.write(f"**Branch:** {trans.get('branch_id', 'N/A')}")
        st.write(f"**Review Type:** {alert_type_display}")

    with col2:
        st.write(f"**Timestamp:** {trans['timestamp']}")
        st.write(f"**ML Probability:** {trans['ml_probability']:.3f}")

        # Color-coded probability indicator
        prob_color = "red" if trans["ml_probability"] > 0.9 else "orange" if trans["ml_probability"] > 0.8 else "yellow"
        st.markdown(
            f"<span style='color:{prob_color}; font-weight:bold;'>Risk Level: {'HIGH' if trans['ml_probability'] > 0.9 else 'MODERATE' if trans['ml_probability'] > 0.8 else 'LOW'}</span>",
            unsafe_allow_html=True,
        )

    # Explainability: top risk factors
    try:
        headers = get_api_headers()
        explain_url = get_api_url(f"alerts/{trans['transaction_id']}/explain")
        explain_response = requests.get(explain_url, headers=headers)
        if explain_response.status_code == 200:
            explanation = explain_response.json()
            shap_top5 = explanation.get("shap_top5", [])
            factors = shap_top5 or explanation.get("top_risk_factors", [])
            if factors:
                st.subheader("SHAP Explainability")
                if shap_top5:
                    shap_df = pd.DataFrame(shap_top5)
                    chart_df = shap_df.sort_values("abs_impact", ascending=True)
                    fig = go.Figure(
                        go.Waterfall(
                            orientation="h",
                            measure=["relative"] * len(chart_df),
                            y=chart_df["description"] if "description" in chart_df else chart_df["feature"],
                            x=chart_df["impact"],
                            text=[f"value={value:.3f}" for value in chart_df["value"]],
                            connector={"line": {"color": "rgba(80,80,80,0.35)"}},
                            increasing={"marker": {"color": "#c2410c"}},
                            decreasing={"marker": {"color": "#0f766e"}},
                        )
                    )
                    fig.update_layout(
                        title="Top 5 SHAP Feature Contributions",
                        xaxis_title="SHAP impact on fraud score",
                        yaxis_title="Feature",
                        height=320,
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                for factor in factors:
                    description = factor.get("description", factor.get("feature", ""))
                    impact = factor.get("impact")
                    if impact is not None:
                        st.write(f"- {description}: value={factor.get('value')}, " f"SHAP impact={impact:.4f}")
                    elif factor.get("score") is not None:
                        st.write(f"- {description} (score: {factor.get('score')})")
                    else:
                        st.write(f"- {description}")
            else:
                st.caption("No explainability data available.")
        else:
            st.caption("Explainability unavailable.")
    except Exception:
        st.caption("Explainability unavailable.")

    # Show SAR report if available
    if trans.get("sar_report") and show_sar:
        st.markdown("---")
        st.subheader("📋 Generated SAR Report")
        st.text_area("SAR Report", value=trans["sar_report"], height=200, disabled=True)

    # Feedback form
    st.markdown("---")
    st.subheader("✅ Analyst Feedback")

    with st.form(key="feedback_form"):
        analyst_label = st.radio("Classification:", options=["Confirmed Fraud", "False Positive"], horizontal=True)

        analyst_comment = st.text_area("Additional Comments (Optional)")

        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            try:
                feedback_payload = {
                    "transaction_id": trans["transaction_id"],
                    "analyst_label": analyst_label,
                    "analyst_comment": analyst_comment,
                    "branch_id": trans.get("branch_id"),
                }

                # Use the shared utility functions
                headers = get_api_headers()
                headers["Content-Type"] = "application/json"
                if user_id:
                    headers["X-User-Id"] = user_id
                api_url = get_api_url("feedback/")

                response = requests.post(api_url, json=feedback_payload, headers=headers)

                if response.status_code == 200:
                    st.success("Feedback submitted successfully!")
                    # Clear the selected transaction to force refresh
                    st.session_state.selected_transaction = None
                else:
                    st.error(f"Failed to submit feedback: {response.status_code}")
            except requests.exceptions.ConnectionError:
                api_url_display = os.getenv("COMMAND_CENTER_API_URL", "http://localhost:8001")
                st.error(
                    f"Could not connect to the API. Please ensure the FastAPI server is running on {api_url_display}."
                )
            except Exception as e:
                st.error(f"Error submitting feedback: {str(e)}")

# Auto-refresh mechanism
time.sleep(refresh_interval)
st.rerun()
