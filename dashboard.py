import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set(style='darkgrid')
plt.style.use('seaborn-v0_8-darkgrid')

# Load data
@st.cache_data
def load_data():
    customers_df = pd.read_csv('https://docs.google.com/spreadsheets/d/13H2D7q2kR6YWw3MsxZM5m9PO8yth5ywVMcydGsZwTAo/export?format=csv')
    orders_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1QF33zpQtpovUaN2feVrGR-fJq5iyEoRrlbvc6MxaZAk/export?format=csv')
    order_items_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1dAMq2cA_rOkk_GLBVyVYxIY7Jy-OY7X7TVXVsarvhNM/export?format=csv')

    # Convert datetime columns
    datetime_columns = ['order_purchase_timestamp', 'order_approved_at',
                        'order_delivered_carrier_date', 'order_delivered_customer_date',
                        'order_estimated_delivery_date']

    for col in datetime_columns:
        if col in orders_df.columns:
            orders_df[col] = pd.to_datetime(orders_df[col], errors='coerce')

    # Missing Value
    for col in ['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date']:
        if col in orders_df.columns:
            orders_df = orders_df.sort_values('order_purchase_timestamp')
            orders_df[col] = orders_df[col].ffill() 
            orders_df[col] = orders_df[col].bfill() 

    return customers_df, orders_df, order_items_df

# Helper functions
def create_monthly_orders_df(orders_df, order_items_df):
    orders_df['order_year'] = orders_df['order_purchase_timestamp'].dt.year
    orders_df['order_month'] = orders_df['order_purchase_timestamp'].dt.month
    orders_df['order_yearmonth'] = orders_df['order_purchase_timestamp'].dt.to_period('M')

    monthly_orders = orders_df.groupby('order_yearmonth').size().reset_index(name='order_count')
    monthly_orders['order_yearmonth_str'] = monthly_orders['order_yearmonth'].astype(str)

    order_revenue = pd.merge(orders_df, order_items_df, on='order_id', how='inner')
    monthly_revenue = order_revenue.groupby('order_yearmonth')['price'].sum().reset_index(name='revenue')

    return monthly_orders, monthly_revenue

def create_rfm_df(orders_df, order_items_df):
    reference_date = orders_df['order_purchase_timestamp'].max()
    rfm_data = pd.merge(orders_df, order_items_df, on='order_id', how='inner')

    rfm_df = rfm_data.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,
        'order_id': 'nunique',
        'price': 'sum'
    }).reset_index()

    rfm_df.columns = ['customer_id', 'recency', 'frequency', 'monetary']

    # RFM scores
    rfm_df['r_score'] = pd.qcut(rfm_df['recency'], 4, labels=['4', '3', '2', '1']).astype(int)
    rfm_df['f_score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), 4, labels=['1', '2', '3', '4']).astype(int)
    rfm_df['m_score'] = pd.qcut(rfm_df['monetary'].rank(method='first'), 4, labels=['1', '2', '3', '4']).astype(int)
    rfm_df['rfm_score'] = rfm_df['r_score'] + rfm_df['f_score'] + rfm_df['m_score']

    # Segment customers
    def get_segment(row):
        if row['r_score'] >= 3 and row['f_score'] >= 3 and row['m_score'] >= 3:
            return 'Best Customer'
        elif row['r_score'] >= 3 and row['f_score'] >= 2:
            return 'Loyal Customer'
        elif row['r_score'] <= 2 and row['f_score'] >= 3:
            return 'At Risk Customer'
        elif row['r_score'] <= 2 and row['f_score'] <= 2:
            return 'Lost Customer'
        else:
            return 'Regular Customer'

    rfm_df['segment'] = rfm_df.apply(get_segment, axis=1)
    return rfm_df

def create_geographic_df(orders_df, order_items_df, customers_df):
    cust_geo = customers_df[['customer_id', 'customer_state', 'customer_city']]
    sales_geo = pd.merge(orders_df, cust_geo, on='customer_id', how='inner')
    sales_geo = pd.merge(sales_geo, order_items_df, on='order_id', how='inner')

    revenue_per_state = sales_geo.groupby('customer_state')['price'].agg(['sum', 'count', 'mean']).reset_index()
    revenue_per_state.columns = ['state', 'total_revenue', 'total_orders', 'avg_order_value']
    revenue_per_state = revenue_per_state.sort_values('total_revenue', ascending=False)

    return revenue_per_state

# Dashboard
st.set_page_config(page_title="E-Commerce Dashboard", page_icon="📊", layout="wide")

with st.spinner('Loading data...'):
    customers_df, orders_df, order_items_df = load_data()

monthly_orders, monthly_revenue = create_monthly_orders_df(orders_df, order_items_df)
rfm_df = create_rfm_df(orders_df, order_items_df)
revenue_per_state = create_geographic_df(orders_df, order_items_df, customers_df)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2838/2838912.png", width=100)
    st.title("E-Commerce Dashboard")
    st.markdown("---")

    min_date = orders_df['order_purchase_timestamp'].min().date()
    max_date = orders_df['order_purchase_timestamp'].max().date()

    date_range = st.date_input(
        "Pilih Rentang Waktu",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    st.markdown("---")
    st.markdown("### Informasi")
    st.markdown(f"- Total Customer: {customers_df['customer_id'].nunique():,}")
    st.markdown(f"- Total Order: {orders_df['order_id'].nunique():,}")
    st.markdown(f"- Periode Data: {min_date} - {max_date}")

# Main content
st.title("📊 E-Commerce Public Dataset Analysis")
st.markdown("### Dashboard Analisis Penjualan dan Perilaku Pelanggan")
st.markdown("---")

# Key Metrics
st.subheader("📈 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = monthly_revenue['revenue'].sum()
    st.metric("Total Revenue", f"${total_revenue:,.2f}")

with col2:
    total_orders = monthly_orders['order_count'].sum()
    st.metric("Total Orders", f"{total_orders:,}")

with col3:
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    st.metric("Average Order Value", f"${avg_order_value:.2f}")

with col4:
    unique_customers = rfm_df['customer_id'].nunique()
    st.metric("Unique Customers", f"{unique_customers:,}")

st.markdown("---")

# Sales Performance
st.subheader("📅 Sales Performance Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_orders['order_yearmonth_str'], monthly_orders['order_count'],
            marker='o', linewidth=2, color='#2E86AB')
    ax.set_title('Jumlah Order per Bulan (2016-2018)', fontsize=12)
    ax.set_xlabel('Periode (Bulan-Tahun)')
    ax.set_ylabel('Jumlah Order')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_revenue['order_yearmonth'].astype(str), monthly_revenue['revenue'],
            marker='s', linewidth=2, color='#A23B72')
    ax.set_title('Total Revenue per Bulan (2016-2018)', fontsize=12)
    ax.set_xlabel('Periode (Bulan-Tahun)')
    ax.set_ylabel('Revenue ($)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# Geographic Sales Analysis
st.subheader("🗺️ Geographic Sales Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    top_states = revenue_per_state.head(8)
    colors = ['#2E86AB'] * len(top_states)
    colors[0] = '#A23B72'
    bars = ax.bar(top_states['state'], top_states['total_revenue'], color=colors)
    ax.set_title('Top 8 State dengan Total Revenue Tertinggi', fontsize=12)
    ax.set_xlabel('State')
    ax.set_ylabel('Total Revenue ($)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars, top_states['total_revenue']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'${value/1000:.0f}K', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(10, 6))
    other_sum = revenue_per_state.iloc[8:]['total_revenue'].sum()
    pie_data = revenue_per_state.head(7)[['state', 'total_revenue']].copy()
    pie_data.loc[len(pie_data)] = ['Lainnya', other_sum]

    colors_pie = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D',
                  '#6A994E', '#BC4A6C', '#5D576B', '#8A817C']

    ax.pie(pie_data['total_revenue'], labels=pie_data['state'],
           autopct='%1.0f%%', colors=colors_pie[:len(pie_data)])
    ax.set_title('Distribusi Revenue per State', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

# Customer Segmentation
st.subheader("👥 Customer Segmentation (RFM Analysis)")

segment_counts = rfm_df['segment'].value_counts()

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 8))
    colors_segment = ['#F4D03F', '#2ECC71', '#E74C3C', '#E67E22', '#3498DB']
    ax.pie(segment_counts.values, labels=segment_counts.index,
           autopct='%1.1f%%', colors=colors_segment, explode=[0.05] * len(segment_counts))
    ax.set_title('Distribusi Segmentasi Pelanggan', fontsize=12)
    st.pyplot(fig)

with col2:
    st.markdown("##### Interpretasi Segmentasi:")
    for segment, count in segment_counts.items():
        percentage = (count / len(rfm_df)) * 100
        st.markdown(f"- **{segment}**: {count:,} pelanggan ({percentage:.1f}%)")

# RFM Distribution
st.subheader("📊 Distribusi RFM")

col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rfm_df['recency'], bins=30, color='#2E86AB', edgecolor='black', alpha=0.7)
    ax.set_title('Distribusi Recency', fontsize=12)
    ax.set_xlabel('Hari Sejak Transaksi Terakhir')
    ax.set_ylabel('Jumlah Customer')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rfm_df['frequency'], bins=20, color='#A23B72', edgecolor='black', alpha=0.7)
    ax.set_title('Distribusi Frequency', fontsize=12)
    ax.set_xlabel('Jumlah Transaksi')
    ax.set_ylabel('Jumlah Customer')
    plt.tight_layout()
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rfm_df['monetary'], bins=30, color='#F18F01', edgecolor='black', alpha=0.7)
    ax.set_title('Distribusi Monetary', fontsize=12)
    ax.set_xlabel('Total Belanja ($)')
    ax.set_ylabel('Jumlah Customer')
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
st.caption("Copyright © 2026 | E-Commerce Public Dataset Analysis Dashboard")