# AI Google Ads Optimizer Streamlit App
# ====================================
# This Streamlit application connects to the Google Ads API to fetch campaign performance
# data, uses OpenAI's language model to generate optimization recommendations,
# and displays actionable insights.

import os
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import openai

# Load credentials from environment
GOOGLE_ADS_CONFIG_PATH = os.getenv("GOOGLE_ADS_CONFIG_PATH", "google-ads.yaml")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
@st.cache_resource
def init_google_ads_client():
    return GoogleAdsClient.load_from_storage(GOOGLE_ADS_CONFIG_PATH)

@st.cache_resource
def init_openai():
    openai.api_key = OPENAI_API_KEY
    return openai


def fetch_performance_data(client, customer_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    query = f"""
        SELECT
          campaign.id,
          campaign.name,
          metrics.impressions,
          metrics.clicks,
          metrics.cost_micros,
          metrics.conversions,
          metrics.ctr
        FROM campaign
        WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    """
    service = client.get_service("GoogleAdsService")
    response = service.search_stream(customer_id=customer_id, query=query)

    records = []
    for batch in response:
        for row in batch.results:
            metrics = row.metrics
n            records.append({
                "campaign_id": row.campaign.id,
                "campaign_name": row.campaign.name,
                "impressions": metrics.impressions,
                "clicks": metrics.clicks,
                "cost": metrics.cost_micros / 1e6,
                "conversions": metrics.conversions,
                "ctr": metrics.ctr,
            })
    return pd.DataFrame(records)


def generate_recommendations(df: pd.DataFrame) -> str:
    # Summarize key metrics
    summary = df.describe().to_string()
    prompt = (
        "You are an experienced Google Ads consultant. Based on the following campaign performance metrics, "
        "provide actionable optimization recommendations. Include suggestions about bids, budgets, keywords, "
        "and ad copy improvements. \n\nMetrics Summary:\n" + summary
    )
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


def format_date_range(days_back: int):
    end = datetime.today().date()
    start = end - timedelta(days=days_back)
    return start.isoformat(), end.isoformat()


# Streamlit UI
st.set_page_config(page_title="AI Google Ads Optimizer", layout="wide")
st.title("🚀 AI Google Ads Optimizer")

client = init_google_ads_client()
openai = init_openai()

with st.sidebar:
    st.header("Settings")
    customer_id = st.text_input("Google Ads Customer ID", help="e.g., 123-456-7890")
    days_back = st.slider("Date Range (days)", min_value=7, max_value=90, value=30, step=7)
    if not OPENAI_API_KEY:
        st.error("Please set OPENAI_API_KEY environment variable.")
    if not os.path.exists(GOOGLE_ADS_CONFIG_PATH):
        st.error("Google Ads config not found at: {}".format(GOOGLE_ADS_CONFIG_PATH))

start_date, end_date = format_date_range(days_back)

if st.button("Fetch & Optimize"):
    if not customer_id:
        st.warning("Enter a Google Ads Customer ID to proceed.")
    else:
        with st.spinner("Fetching performance data..."):
            try:
                df = fetch_performance_data(client, customer_id.replace('-', ''), start_date, end_date)
                st.success("Fetched data for {} campaigns".format(len(df)))
                st.dataframe(df)

                st.subheader("AI-Generated Recommendations")
                with st.spinner("Generating recommendations..."):
                    recs = generate_recommendations(df)
                st.markdown(recs)

            except GoogleAdsException as ex:
                st.error(f"Google Ads API Error: {ex.error.message}")

# Deployment instructions
st.sidebar.markdown("---")
st.sidebar.markdown(
