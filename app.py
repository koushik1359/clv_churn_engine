import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CLV & Churn Engine",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Premium Dark Theme
# ============================================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    /* KPI Card Styling */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
        margin-bottom: 10px;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
    }
    .kpi-card.danger {
        background: linear-gradient(135deg, #f5365c 0%, #f56036 100%);
        box-shadow: 0 8px 32px rgba(245, 54, 92, 0.3);
    }
    .kpi-card.success {
        background: linear-gradient(135deg, #2dce89 0%, #2dcecc 100%);
        box-shadow: 0 8px 32px rgba(45, 206, 137, 0.3);
    }
    .kpi-card.info {
        background: linear-gradient(135deg, #11cdef 0%, #1171ef 100%);
        box-shadow: 0 8px 32px rgba(17, 205, 239, 0.3);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        margin: 8px 0;
    }
    .kpi-label {
        font-size: 0.9rem;
        font-weight: 400;
        color: rgba(255,255,255,0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 10px 0 20px 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2, #f5365c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .main-header p {
        font-size: 1.05rem;
        color: #8898aa;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #32325d;
        border-left: 4px solid #667eea;
        padding-left: 12px;
        margin: 30px 0 15px 0;
    }
    
    /* Divider */
    .gradient-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f5365c, transparent);
        border: none;
        margin: 20px 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    decision = pd.read_csv('decision_matrix.csv')
    features = pd.read_csv('customer_features.csv')
    return decision, features

decision, features = load_data()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## 🎛️ Control Panel")
st.sidebar.markdown("Use the filters below to explore customer segments.")
st.sidebar.markdown("---")

risk_filter = st.sidebar.slider(
    "⚡ Minimum Churn Risk",
    min_value=0.0, max_value=1.0, value=0.0, step=0.05
)

clv_max = float(decision['CLV_90d'].quantile(0.99))
clv_range = st.sidebar.slider(
    "💰 CLV Range (£)",
    min_value=0.0, max_value=clv_max, value=(0.0, clv_max), step=50.0
)

filtered = decision[
    (decision['Churn_Probability'] >= risk_filter) &
    (decision['CLV_90d'] >= clv_range[0]) &
    (decision['CLV_90d'] <= clv_range[1])
]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing:** {len(filtered):,} of {len(decision):,} customers")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>💰 CLV & Churn Intelligence Engine</h1>
    <p>Combining XGBoost Machine Learning with Bayesian CLV to drive £320K+ in revenue retention</p>
</div>
<div class="gradient-divider"></div>
""", unsafe_allow_html=True)

# ============================================================
# KPI CARDS
# ============================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card info">
        <div class="kpi-label">Total Customers</div>
        <div class="kpi-value">{len(filtered):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_churn = filtered['Churn_Probability'].mean()
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg Churn Risk</div>
        <div class="kpi-value">{avg_churn:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    total_clv = filtered['CLV_90d'].sum()
    st.markdown(f"""
    <div class="kpi-card success">
        <div class="kpi-label">Total CLV (90d)</div>
        <div class="kpi-value">£{total_clv:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    total_risk = filtered['Revenue_at_Risk'].sum()
    st.markdown(f"""
    <div class="kpi-card danger">
        <div class="kpi-label">Revenue at Risk</div>
        <div class="kpi-value">£{total_risk:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# STRATEGY MATRIX
# ============================================================
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">📊 Customer Retention Strategy Matrix</div>', unsafe_allow_html=True)

clv_median = decision['CLV_90d'].median()

def assign_quadrant(row):
    if row['Churn_Probability'] >= 0.5 and row['CLV_90d'] >= clv_median:
        return '🚨 Save Now'
    elif row['Churn_Probability'] >= 0.5 and row['CLV_90d'] < clv_median:
        return '❌ Let Go'
    elif row['Churn_Probability'] < 0.5 and row['CLV_90d'] >= clv_median:
        return '⭐ VIP'
    else:
        return '😊 Happy'

filtered = filtered.copy()
filtered['Segment'] = filtered.apply(assign_quadrant, axis=1)

color_map = {
    '🚨 Save Now': '#f5365c',
    '❌ Let Go': '#8898aa',
    '⭐ VIP': '#5e72e4',
    '😊 Happy': '#2dce89'
}

fig = px.scatter(
    filtered,
    x='CLV_90d',
    y='Churn_Probability',
    color='Segment',
    color_discrete_map=color_map,
    size='Revenue_at_Risk',
    size_max=22,
    hover_data={'CustomerID': True, 'Revenue_at_Risk': ':.2f', 'CLV_90d': ':.2f', 'Churn_Probability': ':.2f'},
    range_x=[0, filtered['CLV_90d'].quantile(0.95)],
)

fig.add_hline(y=0.5, line_dash="dot", line_color="#f5365c", opacity=0.6,
              annotation_text="Churn Threshold", annotation_position="top right")
fig.add_vline(x=clv_median, line_dash="dot", line_color="#5e72e4", opacity=0.6,
              annotation_text="CLV Median", annotation_position="top right")

fig.update_layout(
    height=520,
    template='plotly_white',
    font=dict(family='Inter'),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="Customer Lifetime Value (£)",
    yaxis_title="Churn Probability",
)

st.plotly_chart(fig)
# ============================================================
# SEGMENT BREAKDOWN
# ============================================================
st.markdown('<div class="section-header">📈 Segment Breakdown</div>', unsafe_allow_html=True)

seg_col1, seg_col2 = st.columns(2)

with seg_col1:
    seg_counts = filtered['Segment'].value_counts()
    fig_pie = px.pie(
        values=seg_counts.values, names=seg_counts.index,
        color=seg_counts.index, color_discrete_map=color_map,
        hole=0.45
    )
    fig_pie.update_layout(
        font=dict(family='Inter'),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie)

with seg_col2:
    seg_revenue = filtered.groupby('Segment')['Revenue_at_Risk'].sum().sort_values(ascending=True)
    fig_bar = px.bar(
        x=seg_revenue.values, y=seg_revenue.index, orientation='h',
        color=seg_revenue.index, color_discrete_map=color_map,
        labels={'x': 'Revenue at Risk (£)', 'y': ''}
    )
    fig_bar.update_layout(
        showlegend=False,
        font=dict(family='Inter'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350
    )
    fig_bar.update_traces(marker_line_width=0)
    st.plotly_chart(fig_bar)

# ============================================================
# TOP AT-RISK CUSTOMERS
# ============================================================
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">🔥 Priority Retention List</div>', unsafe_allow_html=True)
st.caption("These customers should receive immediate retention offers based on their combined churn risk and lifetime value.")

top_n = st.slider("Number of customers to display", 5, 50, 15)
top_risk = filtered.nlargest(top_n, 'Revenue_at_Risk')[
    ['CustomerID', 'Segment', 'Churn_Probability', 'CLV_90d', 'Revenue_at_Risk']
].reset_index(drop=True)
top_risk.index += 1
top_risk.columns = ['Customer ID', 'Segment', 'Churn Risk', 'CLV (£)', 'Revenue at Risk (£)']

st.dataframe(
    top_risk.style.format({
        'Churn Risk': '{:.1%}',
        'CLV (£)': '£{:,.2f}',
        'Revenue at Risk (£)': '£{:,.2f}'
    }).background_gradient(subset=['Revenue at Risk (£)'], cmap='OrRd'),
    height=450
)


# ============================================================
# FOOTER
# ============================================================
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 15px 0; color: #8898aa; font-size: 0.85rem;'>
    <strong>CLV & Churn Intelligence Engine</strong><br>
    Built with XGBoost · BG/NBD · Gamma-Gamma · Streamlit · Plotly<br>
    <span style='font-size: 0.75rem;'>Data: UK E-Commerce Dataset (540K+ Transactions) | © 2026</span>
</div>
""", unsafe_allow_html=True)
