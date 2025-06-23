import os
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import openai
import yaml

# Debug: inspect your config file
config_path = os.getenv("GOOGLE_ADS_CONFIG_PATH", "google-ads.yaml")
try:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    st.sidebar.write("🔍 Loaded google-ads.yaml keys:", list(cfg.keys()))
    sample_values = {k: ("…" if k == "refresh_token" else v) for k, v in cfg.items()}
    st.sidebar.write("🔍 Sample values:", sample_values)
except Exception as e:
    st.sidebar.error(f"Failed to read {config_path}: {e}")
    st.stop()

# Load credentials from environment or default YAML
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

# Fetch basic campaign performance
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
            m = row.metrics
            records.append({
                "campaign_id": row.campaign.id,
                "campaign_name": row.campaign.name,
                "date": row.segments.date,
                "impressions": m.impressions,
                "clicks": m.clicks,
                "cost": m.cost_micros / 1e6,
                "conversions": m.conversions,
                "ctr": m.ctr,
            })
    return pd.DataFrame(records)

# Fetch search term performance
def fetch_search_terms(client, customer_id: str, date: str) -> pd.DataFrame:
    query = f"""
        SELECT
          segments.search_term,
          metrics.impressions,
          metrics.clicks,
          metrics.cost_micros
        FROM search_term_view
        WHERE segments.date = '{date}'
    """
    service = client.get_service("GoogleAdsService")
    response = service.search_stream(customer_id=customer_id, query=query)

    records = []
    for batch in response:
        for row in batch.results:
            m = row.metrics
            records.append({
                "search_term": row.segments.search_term,
                "impressions": m.impressions,
                "clicks": m.clicks,
                "cost": m.cost_micros / 1e6,
            })
    return pd.DataFrame(records)

# Generate AI recommendations
def generate_recommendations(df: pd.DataFrame) -> str:
    summary = df.describe().to_string()
    prompt = (
        "You are an expert Google Ads consultant. Based on these metrics, "
        "provide optimization opportunities around bids, budgets, keywords, ad copy, and schedule.\n\n" + summary
    )
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()

# Date helpers
def format_date_range(days_back: int):
    today = datetime.today().date()
    start = today - timedelta(days=days_back)
    return start.isoformat(), today.isoformat()

# Streamlit UI
st.set_page_config(page_title="AI Google Ads Optimizer", layout="wide")
st.title("🚀 AI Google Ads Optimizer")

# Sidebar inputs
with st.sidebar:
    st.header("Settings")
    customer_id = st.text_input("Google Ads Customer ID", help="e.g., 123-456-7890")
    days_back = st.slider("History (days)", 7, 90, 30, 7)
    st.markdown("---")
    st.subheader("Manual Conversion Event")
    conv_date = st.date_input("Conversion Date")
    conv_time = st.time_input("Conversion Time")
    conv_location = st.text_input("Conversion Location")
    st.markdown("---")
    if not OPENAI_API_KEY:
        st.error("Set OPENAI_API_KEY in your environment.")
    if not os.path.exists(GOOGLE_ADS_CONFIG_PATH):
        st.error(f"Missing Google Ads config at {GOOGLE_ADS_CONFIG_PATH}")

start_date, end_date = format_date_range(days_back)

if st.button("Fetch & Analyze"):
    if not customer_id:
        st.warning("Enter a Google Ads Customer ID.")
    else:
        client = init_google_ads_client()
        openai = init_openai()
        cid = customer_id.replace('-', '')
        try:
            perf_df = fetch_performance_data(client, cid, start_date, end_date)
            st.subheader("Campaign Performance History")
            st.dataframe(perf_df)

            st.subheader(f"Search Terms on {conv_date} at {conv_location}")
            search_df = fetch_search_terms(client, cid, conv_date.isoformat())
            top_kw = search_df.sort_values('clicks', ascending=False).head(10)
            st.dataframe(top_kw)

            st.subheader("Optimization Opportunities")
            recs = generate_recommendations(perf_df)
            st.markdown(recs)

        except GoogleAdsException as ex:
            st.error(f"API Error: {ex.error.message}")
