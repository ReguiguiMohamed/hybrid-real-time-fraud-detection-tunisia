# dashboard/dashboard.py
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import time

# Add the src directory to the path to import shared utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from shared.utils import get_api_headers, get_api_url

# Set page config
st.set_page_config(
    page_title="Tunisian Fraud Detection - Command Center",
    page_icon="🚨",
    layout="wide"
)

# Title
st.title("🚨 Tunisian Fraud Detection - Operational Command Center")
st.markdown("---")

# Initialize session state
if 'selected_transaction' not in st.session_state:
    st.session_state.selected_transaction = None
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0

# Sidebar for navigation and filters
with st.sidebar:
    st.header("📊 Dashboard Controls")

    # Refresh interval selection
    refresh_interval = st.selectbox(
        "Refresh Interval (seconds)",
        options=[5, 10, 30, 60],
        index=1
    )

    # Filter options
    st.subheader("Filters")
    min_prob = st.slider("Minimum ML Probability (High Risk)", 0.0, 1.0, 0.85)
    show_sar = st.checkbox("Show SAR Reports", value=True)
    alert_type_filter = st.selectbox(
        "Alert Type",
        options=["All", "High Risk", "Random Sample"],
        index=0
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

        if alert_type_param:
            api_url = get_api_url(f"alerts/review-queue/?limit=50&alert_type={alert_type_param}")
        else:
            api_url = get_api_url("alerts/review-queue/?limit=50")

        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            alerts = response.json()

            if alerts:
                # Convert to DataFrame for easier manipulation
                df = pd.DataFrame(alerts)

                if 'alert_type' not in df.columns:
                    df['alert_type'] = "high_risk"
                df['alert_type'] = df['alert_type'].fillna("high_risk")

                # Filter by minimum probability for high-risk alerts only
                keep_mask = (df['alert_type'] != "high_risk") | (df['ml_probability'] >= min_prob)
                df_filtered = df[keep_mask]

                if not df_filtered.empty:
                    df_display = df_filtered.copy()
                    df_display['alert_type_display'] = df_display['alert_type'].map({
                        "high_risk": "High Risk",
                        "random_sample": "Random Sample"
                    }).fillna("High Risk")

                    # Display alerts in a table
                    df_table = df_display[['transaction_id', 'user_id', 'amount_tnd', 'governorate',
                                           'payment_method', 'ml_probability', 'alert_type_display']].rename(
                        columns={"alert_type_display": "alert_type"}
                    )
                    st.dataframe(
                        df_table.style.format({
                            'amount_tnd': '{:.2f}',
                            'ml_probability': '{:.3f}'
                        }),
                        use_container_width=True,
                        height=400
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
                            )
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

        stats_response = requests.get(api_url, headers=headers)
        if stats_response.status_code == 200:
            stats = stats_response.json()

            # Display key metrics
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Total Feedback", stats.get('total_feedback', 0))
            with col2_2:
                st.metric("Alert Precision", f"{stats.get('precision', 0):.3f}")

            random_sample_fraud_rate = stats.get("random_sample_fraud_rate")
            if random_sample_fraud_rate is not None:
                st.metric("Random Sample Fraud Rate", f"{random_sample_fraud_rate:.3f}")

            st.metric("High-Risk Alerts", stats.get('high_risk_alerts', 0))
            st.metric("Random Samples", stats.get('random_sample_alerts', 0))
            st.metric("Review Queue Total", stats.get('review_queue_total', 0))
            if stats.get('random_sample_rate') is not None:
                st.caption(f"Random sample rate: {stats.get('random_sample_rate'):.3f}")

            # Feedback breakdown chart
            if stats.get('feedback_breakdown'):
                breakdown = stats['feedback_breakdown']
                breakdown_df = pd.DataFrame(list(breakdown.items()), columns=['Label', 'Count'])

                fig = px.pie(breakdown_df, values='Count', names='Label', title='Feedback Distribution')
                st.plotly_chart(fig, use_container_width=True)


            perf_url = get_api_url("monitoring/model-performance/")
            perf_response = requests.get(perf_url, headers=headers)
            if perf_response.status_code == 200:
                perf = perf_response.json()
                st.subheader("Model Performance (Sampling-Aware)")
                st.metric("Estimated Recall", f"{perf.get('recall', 0):.3f}")
                st.metric("Reviewed Recall", f"{perf.get('reviewed_recall', 0):.3f}")
                st.metric("Estimated False Negatives", perf.get('estimated_false_negatives', 0))
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
    alert_type = trans.get('alert_type', 'high_risk')
    alert_type_display = "Random Sample" if alert_type == "random_sample" else "High Risk"

    if alert_type == "random_sample":
        st.info("Random sample review. Use 'False Positive' for non-fraud cases.")

    # Display transaction details
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Transaction ID:** {trans['transaction_id']}")
        st.write(f"**User ID:** {trans['user_id']}")
        st.write(f"**Amount:** {trans['amount_tnd']:.2f} TND")
        st.write(f"**Governorate:** {trans['governorate']}")
        st.write(f"**Payment Method:** {trans['payment_method']}")
        st.write(f"**Review Type:** {alert_type_display}")

    with col2:
        st.write(f"**Timestamp:** {trans['timestamp']}")
        st.write(f"**ML Probability:** {trans['ml_probability']:.3f}")

        # Color-coded probability indicator
        prob_color = "red" if trans['ml_probability'] > 0.9 else "orange" if trans['ml_probability'] > 0.8 else "yellow"
        st.markdown(f"<span style='color:{prob_color}; font-weight:bold;'>Risk Level: {'HIGH' if trans['ml_probability'] > 0.9 else 'MODERATE' if trans['ml_probability'] > 0.8 else 'LOW'}</span>", unsafe_allow_html=True)

    # Show SAR report if available
    if trans.get('sar_report') and show_sar:
        st.markdown("---")
        st.subheader("📋 Generated SAR Report")
        st.text_area("SAR Report", value=trans['sar_report'], height=200, disabled=True)

    # Feedback form
    st.markdown("---")
    st.subheader("✅ Analyst Feedback")

    with st.form(key='feedback_form'):
        analyst_label = st.radio(
            "Classification:",
            options=["Confirmed Fraud", "False Positive"],
            horizontal=True
        )

        analyst_comment = st.text_area("Additional Comments (Optional)")

        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            try:
                feedback_payload = {
                    "transaction_id": trans['transaction_id'],
                    "analyst_label": analyst_label,
                    "analyst_comment": analyst_comment
                }

                # Use the shared utility functions
                headers = get_api_headers()
                headers["Content-Type"] = "application/json"
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
                st.error(f"Could not connect to the API. Please ensure the FastAPI server is running on {api_url_display}.")
            except Exception as e:
                st.error(f"Error submitting feedback: {str(e)}")

# Auto-refresh mechanism
time.sleep(refresh_interval)
st.rerun()
