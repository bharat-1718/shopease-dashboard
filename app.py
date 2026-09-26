import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import statsmodels.formula.api as smf

# --- 1. Dashboard Configuration ---
st.set_page_config(page_title="Executive Analytics", layout="wide", page_icon="📈")
SEED = 61729198
SAMPLE_TARGET = 2501

# --- 2. Clean First, Sample Second Pipeline ---
# Renamed function to instantly break Streamlit's stubborn cache
@st.cache_data
def load_dataset_v3():
    try:
        df = pd.read_csv("shopease_raw_orders.csv")
    except FileNotFoundError:
        st.error("Please place 'shopease_raw_orders.csv' in the folder.")
        st.stop()

    # STEP 1: Remove duplicate Order IDs
    if "OrderID" in df.columns:
        df = df.drop_duplicates(subset="OrderID", keep="first")

    # STEP 2: Standardize text and force bad text to NaN
    for col in ["Gender", "City", "Category", "PaymentMethod", "OrderStatus"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].replace(["Nan", "None", "Null", "", "Na"], np.nan)

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"F": "Female", "M": "Male", "Female": "Female", "Male": "Male"})
    if "OrderStatus" in df.columns:
        df["OrderStatus"] = df["OrderStatus"].replace({"Canceled": "Cancelled"})

    # STEP 3: Standardize numbers and force outliers to NaN
    def parse_discount(x):
        if pd.isna(x): return np.nan
        x = str(x).strip()
        if x.lower() in ['nan', 'none', 'null', '']: return np.nan
        return float(x.replace("%", "")) / 100 if x.endswith("%") else float(x)
    
    if "Discount" in df.columns:
        df["Discount"] = df["Discount"].apply(parse_discount).astype(float)

    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").astype(float)
        df.loc[df["Quantity"] < 1, "Quantity"] = np.nan

    if "CustomerAge" in df.columns:
        df["CustomerAge"] = pd.to_numeric(df["CustomerAge"], errors="coerce").astype(float)
        df.loc[(df["CustomerAge"] < 10) | (df["CustomerAge"] > 100), "CustomerAge"] = np.nan

    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").astype(float)
        df.loc[(df["Rating"] < 1) | (df["Rating"] > 5), "Rating"] = np.nan

    if "UnitPrice" in df.columns and "TotalAmount" in df.columns:
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").astype(float)
        df["TotalAmount"] = pd.to_numeric(df["TotalAmount"], errors="coerce").astype(float)
        
        # Check for mathematically impossible prices
        implied_total = df["Quantity"] * df["UnitPrice"] * (1 - df["Discount"])
        ratio = (implied_total / df["TotalAmount"].replace(0, np.nan)).abs()
        bad_price = (df["UnitPrice"] < 0) | (ratio > 3) | (ratio < 1/3)
        
        # CRITICAL FIX: Use np.where to build a brand new array instead of .loc assignment
        # This completely bypasses Pandas' strict integer/float assignment rules
        corrected_prices = (df["TotalAmount"] / (df["Quantity"] * (1 - df["Discount"]))).round(2)
        df["UnitPrice"] = np.where(bad_price, corrected_prices, df["UnitPrice"])

    # STEP 4: Aggressively drop ANY row that still has a missing value
    df = df.dropna()

    # STEP 5: Sample exactly 2,501 records from the purely clean dataset
    if len(df) < SAMPLE_TARGET:
        st.warning(f"Only {len(df)} fully clean rows remain in the raw file, which is less than {SAMPLE_TARGET}.")
        final_df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    else:
        final_df = df.sample(n=SAMPLE_TARGET, random_state=SEED).reset_index(drop=True)
        
    return final_df

df = load_dataset_v3()

# --- 3. Sidebar Filtering ---
with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    st.caption(f"Target Sample: **{len(df)}** | Seed: **{SEED}**")
    
    categories = st.multiselect("Filter by Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    cities = st.multiselect("Filter by City", sorted(df["City"].unique()), default=sorted(df["City"].unique()))
    
    filtered_df = df[(df["Category"].isin(categories)) & (df["City"].isin(cities))]
    delivered_df = filtered_df[filtered_df["OrderStatus"] == "Delivered"]

# --- 4. Main Executive UI ---
st.title("📈 Executive Commercial Operations")
st.markdown("Strictly cleaned and randomly sampled dataset featuring dynamic visualization and econometric causality.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Records", f"{len(filtered_df):,}", "Exactly 2,501 loaded")
m2.metric("Total Generated Revenue", f"${filtered_df['TotalAmount'].sum():,.0f}")
m3.metric("Average Transaction Value", f"${filtered_df['TotalAmount'].mean():,.2f}")
m4.metric("Mean Customer Rating", f"{filtered_df['Rating'].mean():.2f} / 5.0")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Interactive Analytics", "📋 Dataset Manager", "🔬 Statistical Insights", "📐 Econometric Models"])

# --- TAB 1: INTERACTIVE PLOTLY CHARTS ---
with tab1:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        category_rev = delivered_df.groupby("Category")["TotalAmount"].sum().reset_index()
        fig_bar = px.bar(category_rev, x="Category", y="TotalAmount", text_auto='.2s', 
                         title="Total Revenue by Category (Delivered)", color="Category", template="plotly_white")
        fig_bar.update_traces(textfont_size=12, textangle=0, textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        status_counts = filtered_df["OrderStatus"].value_counts().reset_index()
        fig_donut = px.pie(status_counts, values="count", names="OrderStatus", hole=0.4, 
                           title="Order Fulfillment Status", template="plotly_white")
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_donut, use_container_width=True)

    if not delivered_df.empty:
        fig_scatter = px.scatter(
            delivered_df, 
            x="CustomerAge", 
            y="TotalAmount", 
            color="Category", 
            size="Quantity",
            hover_data=["City", "PaymentMethod"], 
            title="Transaction Analysis: Age vs Value", 
            template="plotly_white", 
            opacity=0.7
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 2: DATASET MANAGER ---
with tab2:
    st.subheader("Cleaned Operational Data (Target: 2501)")
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
    st.subheader("Numeric Summary")
    st.dataframe(filtered_df[["CustomerAge", "Quantity", "UnitPrice", "Discount", "TotalAmount", "Rating"]].describe().T, use_container_width=True)

# --- TAB 3: STATISTICAL INSIGHTS ---
with tab3:
    st.subheader("Behavioral & Statistical Testing")
    c1, c2 = st.columns(2)
    
    with c1:
        with st.expander("Gender Revenue Variance (Welch's T-Test)", expanded=True):
            m_rev = delivered_df.loc[delivered_df["Gender"] == "Male", "TotalAmount"]
            f_rev = delivered_df.loc[delivered_df["Gender"] == "Female", "TotalAmount"]
            
            if not m_rev.empty and not f_rev.empty:
                t_stat, p_val = stats.ttest_ind(m_rev, f_rev, equal_var=False)
                st.metric("T-Statistic", f"{t_stat:.4f}")
                st.metric("P-Value", f"{p_val:.4g}")
            else:
                st.info("Insufficient data for gender testing.")

    with c2:
        with st.expander("Category Revenue Variance (ANOVA)", expanded=True):
            groups = [delivered_df.loc[delivered_df["Category"] == c, "TotalAmount"] for c in delivered_df["Category"].unique()]
            valid_groups = [g for g in groups if len(g) > 0]
            if len(valid_groups) > 1:
                f_stat, p_anova = stats.f_oneway(*valid_groups)
                st.metric("F-Statistic", f"{f_stat:.4f}")
                st.metric("P-Value", f"{p_anova:.4g}")

# --- TAB 4: ECONOMETRIC MODELS ---
with tab4:
    st.subheader("Causal Regressions")
    
    model_choice = st.selectbox("Select Model Architecture", ["OLS: Revenue Drivers", "Logistic: Delivery Probability"])
    
    if model_choice == "OLS: Revenue Drivers":
        if not delivered_df.empty:
            try:
                m = smf.ols("TotalAmount ~ Quantity + UnitPrice + Discount + CustomerAge + C(Category)", data=delivered_df).fit()
                st.code(m.summary().as_text(), language="text")
            except Exception as e:
                st.error(f"Model error: {e}")
            
    elif model_choice == "Logistic: Delivery Probability":
        log_data = filtered_df.copy()
        log_data["Success"] = (log_data["OrderStatus"] == "Delivered").astype(int)
        
        if not log_data.empty:
            try:
                m_log = smf.logit("Success ~ CustomerAge + Discount", data=log_data).fit(disp=False)
                st.code(m_log.summary().as_text(), language="text")
            except Exception as e:
                st.error(f"Convergence error: {e}")