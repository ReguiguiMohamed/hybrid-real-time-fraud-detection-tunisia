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
    min_prob = st.slider("Minimum ML Probability", 0.0, 1.0, 0.85)
    show_sar = st.checkbox("Show SAR Reports", value=True)

    # Stats refresh button
    if st.button("🔄 Refresh Statistics"):
        st.session_state.refresh_counter += 1
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 High-Risk Transaction Alerts")

    # Fetch alerts from API
    try:
        # Get API URL from environment variable, default to localhost
        api_url = os.getenv("COMMAND_CENTER_API_URL", "http://localhost:8001")
        api_token = os.getenv("COMMAND_CENTER_API_TOKEN")

        headers = {}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        response = requests.get(f"{api_url}/alerts/high-risk/?limit=50", headers=headers)
        if response.status_code == 200:
            alerts = response.json()

            if alerts:
                # Convert to DataFrame for easier manipulation
                df = pd.DataFrame(alerts)

                # Filter by minimum probability
                df_filtered = df[df['ml_probability'] >= min_prob]

                if not df_filtered.empty:
                    # Display alerts in a table
                    st.dataframe(
                        df_filtered[['transaction_id', 'user_id', 'amount_tnd', 'governorate',
                                    'payment_method', 'ml_probability']].style.format({
                                        'amount_tnd': '{:.2f}',
                                        'ml_probability': '{:.3f}'
                                    }),
                        use_container_width=True,
                        height=400
                    )

                    # Show transaction details when selected
                    if len(df_filtered) > 0:
                        selected_idx = st.selectbox(
                            "Select Transaction for Details",
                            options=range(len(df_filtered)),
                            format_func=lambda x: f"{df_filtered.iloc[x]['transaction_id']} - {df_filtered.iloc[x]['amount_tnd']:.2f} TND"
                        )

                        if selected_idx is not None:
                            selected_row = df_filtered.iloc[selected_idx]
                            st.session_state.selected_transaction = selected_row.to_dict()
                else:
                    st.info("No high-risk alerts matching current filters.")
            else:
                st.info("No high-risk alerts available.")
        else:
            st.error(f"Failed to fetch alerts: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the API. Please ensure the FastAPI server is running on {api_url}")
    except Exception as e:
        st.error(f"Error fetching alerts: {str(e)}")

with col2:
    st.subheader("📊 System Statistics")

    # Fetch stats from API
    try:
        # Get API URL from environment variable, default to localhost
        api_url = os.getenv("COMMAND_CENTER_API_URL", "http://localhost:8001")
        api_token = os.getenv("COMMAND_CENTER_API_TOKEN")

        headers = {}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        stats_response = requests.get(f"{api_url}/stats", headers=headers)
        if stats_response.status_code == 200:
            stats = stats_response.json()

            # Display key metrics
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Total Feedback", stats.get('total_feedback', 0))
            with col2_2:
                st.metric("Precision", f"{stats.get('precision', 0):.3f}")

            st.metric("High-Risk Alerts", stats.get('high_risk_alerts', 0))

            # Feedback breakdown chart
            if stats.get('feedback_breakdown'):
                breakdown = stats['feedback_breakdown']
                breakdown_df = pd.DataFrame(list(breakdown.items()), columns=['Label', 'Count'])

                fig = px.pie(breakdown_df, values='Count', names='Label', title='Feedback Distribution')
                st.plotly_chart(fig, use_container_width=True)

                # Show monitoring stats
                if 'monitoring_stats' in stats:
                    monitoring = stats['monitoring_stats']
                    st.subheader("🔍 Monitoring Stats")
                    st.write(f"Feedback Processed: {monitoring.get('feedback_processed', 0)}")
                    st.write(f"Confirmed Fraud: {monitoring.get('confirmed_fraud_count', 0)}")
                    st.write(f"False Positives: {monitoring.get('false_positive_count', 0)}")
        else:
            st.error(f"Failed to fetch stats: {stats_response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the API for statistics on {api_url}.")
    except Exception as e:
        st.error(f"Error fetching stats: {str(e)}")

# Detailed view for selected transaction
if st.session_state.selected_transaction:
    st.markdown("---")
    st.subheader("🔍 Transaction Details")

    trans = st.session_state.selected_transaction

    # Display transaction details
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Transaction ID:** {trans['transaction_id']}")
        st.write(f"**User ID:** {trans['user_id']}")
        st.write(f"**Amount:** {trans['amount_tnd']:.2f} TND")
        st.write(f"**Governorate:** {trans['governorate']}")
        st.write(f"**Payment Method:** {trans['payment_method']}")

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

                # Get API URL from environment variable, default to localhost
                api_url = os.getenv("COMMAND_CENTER_API_URL", "http://localhost:8001")
                api_token = os.getenv("COMMAND_CENTER_API_TOKEN")

                headers = {"Content-Type": "application/json"}
                if api_token:
                    headers["Authorization"] = f"Bearer {api_token}"

                response = requests.post(f"{api_url}/feedback/", json=feedback_payload, headers=headers)

                if response.status_code == 200:
                    st.success("Feedback submitted successfully!")
                    # Clear the selected transaction to force refresh
                    st.session_state.selected_transaction = None
                else:
                    st.error(f"Failed to submit feedback: {response.status_code}")
            except requests.exceptions.ConnectionError:
                api_url = os.getenv("COMMAND_CENTER_API_URL", "http://localhost:8001")
                st.error(f"Could not connect to the API. Please ensure the FastAPI server is running on {api_url}.")
            except Exception as e:
                st.error(f"Error submitting feedback: {str(e)}")

# Auto-refresh mechanism
time.sleep(refresh_interval)
st.rerun()