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
@st.cache_data
def load_dataset_v11():
    try:
        df = pd.read_csv("shopease_raw_orders.csv")
    except FileNotFoundError:
        st.error("Please place 'shopease_raw_orders.csv' in the folder.")
        st.stop()

    if "OrderID" in df.columns:
        df = df.drop_duplicates(subset="OrderID", keep="first")

    for col in ["Gender", "City", "Category", "Product", "PaymentMethod", "OrderStatus"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].replace(["Nan", "None", "Null", "", "Na"], np.nan)

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"F": "Female", "M": "Male", "Female": "Female", "Male": "Male"})
    
    if "OrderStatus" in df.columns:
        df["OrderStatus"] = df["OrderStatus"].replace({
            "Canceled": "Cancelled", 
            "Cancel": "Cancelled",
            "Cencelled": "Cancelled"
        })

    if "OrderDate" in df.columns:
        df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
        fallback_date = df["OrderDate"].mode()[0] if not df["OrderDate"].mode().empty else pd.Timestamp('2023-01-01')
        df["OrderDate"] = df["OrderDate"].fillna(fallback_date)
        df["Month"] = df["OrderDate"].dt.strftime('%Y-%m')

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
        # CRITICAL FIX: Give Cancelled/Pending orders a 0.0 rating so they aren't deleted
        df["Rating"] = df["Rating"].fillna(0.0)

    if "UnitPrice" in df.columns and "TotalAmount" in df.columns:
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").astype(float)
        df["TotalAmount"] = pd.to_numeric(df["TotalAmount"], errors="coerce").astype(float)
        
        implied_total = df["Quantity"] * df["UnitPrice"] * (1 - df["Discount"])
        ratio = (implied_total / df["TotalAmount"].replace(0, np.nan)).abs()
        bad_price = (df["UnitPrice"] < 0) | (ratio > 3) | (ratio < 1/3)
        
        corrected_prices = (df["TotalAmount"] / (df["Quantity"] * (1 - df["Discount"]))).round(2)
        df["UnitPrice"] = np.where(bad_price, corrected_prices, df["UnitPrice"])

    if "DeliveryDate" in df.columns:
        df["DeliveryDate"] = df["DeliveryDate"].fillna("N/A")

    # Now when we dropna, the Cancelled orders will survive because their blank ratings are now 0.0
    df = df.dropna()

    if len(df) < SAMPLE_TARGET:
        st.warning(f"Only {len(df)} clean rows remain, which is less than {SAMPLE_TARGET}.")
        final_df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    else:
        final_df = df.sample(n=SAMPLE_TARGET, random_state=SEED).reset_index(drop=True)
        
    return final_df

df = load_dataset_v11()

# --- 3. Sidebar Filtering ---
with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    st.caption(f"Target Sample: **{len(df)}** | Seed: **{SEED}**")
    st.divider()
    
    st.subheader("Categorical Filters")
    categories = st.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    
    available_products = df[df["Category"].isin(categories)]["Product"].unique() if categories else df["Product"].unique()
    products = st.multiselect("Product", sorted(available_products), default=sorted(available_products))
    
    cities = st.multiselect("City", sorted(df["City"].unique()), default=sorted(df["City"].unique()))
    genders = st.multiselect("Gender", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique()))
    payments = st.multiselect("Payment Method", sorted(df["PaymentMethod"].unique()), default=sorted(df["PaymentMethod"].unique()))
    statuses = st.multiselect("Order Status", sorted(df["OrderStatus"].unique()), default=sorted(df["OrderStatus"].unique()))
    
    st.divider()
    st.subheader("Numeric Filters")
    
    min_age_val = int(df["CustomerAge"].min())
    max_age_val = int(df["CustomerAge"].max())
    age_range = st.slider("Customer Age Range", min_value=min_age_val, max_value=max_age_val, value=(min_age_val, max_age_val), step=1)
    
    # Updated slider to include 0.0 so Cancelled/Pending orders show up
    min_rating, max_rating = st.slider("Customer Rating (0 = Unrated)", min_value=0.0, max_value=5.0, value=(0.0, 5.0), step=0.5)
    
    filtered_df = df[
        (df["Category"].isin(categories)) & 
        (df["Product"].isin(products)) &
        (df["City"].isin(cities)) &
        (df["Gender"].isin(genders)) &
        (df["PaymentMethod"].isin(payments)) &
        (df["OrderStatus"].isin(statuses)) &
        (df["CustomerAge"] >= age_range[0]) & 
        (df["CustomerAge"] <= age_range[1]) &
        (df["Rating"] >= min_rating) & 
        (df["Rating"] <= max_rating)
    ]
    
    delivered_df = filtered_df[filtered_df["OrderStatus"] == "Delivered"]

# --- 4. Main Executive UI ---
st.title("📈 Executive Commercial Operations")
st.markdown("Strictly cleaned dataset featuring interactive visualizations aligned with core charting theory.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Records", f"{len(filtered_df):,}", "Exactly 2,501 loaded")
m2.metric("Total Generated Revenue", f"${filtered_df['TotalAmount'].sum():,.0f}")
m3.metric("Average Transaction Value", f"${filtered_df['TotalAmount'].mean():,.2f}")
# Ensure 0.0 ratings don't drag down the real average
m4.metric("Mean Customer Rating", f"{filtered_df[filtered_df['Rating'] > 0]['Rating'].mean():.2f} / 5.0")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Interactive Analytics", "📋 Dataset Manager", "🔬 Statistical Insights", "📐 Econometric Models"])

# --- TAB 1: THE 6 COURSE-ALIGNED CHARTS ---
with tab1:
    # ROW 1: Bar Plot & Pie Plot
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        # 1. BAR PLOT
        category_rev = delivered_df.groupby("Category")["TotalAmount"].sum().reset_index()
        fig_bar = px.bar(category_rev, x="Category", y="TotalAmount", text_auto='.2s', 
                         title="1. Bar Plot: Revenue by Category", color="Category", template="plotly_white")
        fig_bar.update_traces(textfont_size=12, textangle=0, textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    with r1c2:
        # 2. PIE PLOT (Donut style)
        status_counts = filtered_df["OrderStatus"].value_counts().reset_index()
        fig_pie = px.pie(status_counts, values="count", names="OrderStatus", hole=0.4, 
                           title="2. Pie Plot: Order Status Distribution", template="plotly_white", color="OrderStatus")
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    # ROW 2: Line Plot & Histogram
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        # 3. LINE PLOT
        if "Month" in delivered_df.columns and not delivered_df.empty:
            trend_df = delivered_df.groupby("Month")["TotalAmount"].sum().reset_index().sort_values("Month")
            fig_line = px.line(trend_df, x="Month", y="TotalAmount", markers=True,
                               title="3. Line Plot: Monthly Revenue Trend", template="plotly_white")
            st.plotly_chart(fig_line, use_container_width=True)

    with r2c2:
        # 4. HISTOGRAM PLOT
        fig_hist = px.histogram(filtered_df, x="CustomerAge", nbins=15, 
                                title="4. Histogram: Customer Age Distribution", 
                                template="plotly_white", color_discrete_sequence=["#1f77b4"])
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ROW 3: Box Plot & Scatter Plot
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        # 5. BOX PLOT (Simplified: Age vs Category)
        if not filtered_df.empty:
            fig_box = px.box(filtered_df, x="Category", y="CustomerAge", color="Category",
                             title="5. Box Plot: Customer Age Spread by Category", template="plotly_white")
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

    with r3c2:
        # 6. SCATTER PLOT (Simplified: Clean dots, no overlapping bubbles)
        if not filtered_df.empty:
            fig_scatter = px.scatter(
                filtered_df, 
                x="Quantity", 
                y="TotalAmount", 
                color="Category", 
                hover_data=["City", "PaymentMethod", "Gender", "Product"], 
                title="6. Scatter Plot: Quantity vs. Total Revenue", 
                template="plotly_white", 
                opacity=0.7
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 2, 3, 4: RETAINED ANALYSIS ---
with tab2:
    st.subheader("Cleaned Operational Data (Target: 2501)")
    st.dataframe(filtered_df, use_container_width=True, height=400)
    st.subheader("Numeric Summary")
    st.dataframe(filtered_df[["CustomerAge", "Quantity", "UnitPrice", "Discount", "TotalAmount", "Rating"]].describe().T, use_container_width=True)

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
                st.info("Insufficient data.")
    with c2:
        with st.expander("Category Revenue Variance (ANOVA)", expanded=True):
            groups = [delivered_df.loc[delivered_df["Category"] == c, "TotalAmount"] for c in delivered_df["Category"].unique()]
            valid_groups = [g for g in groups if len(g) > 0]
            if len(valid_groups) > 1:
                f_stat, p_anova = stats.f_oneway(*valid_groups)
                st.metric("F-Statistic", f"{f_stat:.4f}")
                st.metric("P-Value", f"{p_anova:.4g}")

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
