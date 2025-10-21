import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import numpy as np

st.set_page_config(page_title="NBA Draft vs Performance", layout="wide")
st.title("🏀 NBA Draft Position vs Current Season Performance")
st.markdown("Analyzing how draft position correlates with 2024-25 season performance")

@st.cache_data
def fetch_draft_data():
    """Scrape 2024 NBA draft data from Basketball Reference"""
    url = "https://www.basketball-reference.com/draft/NBA_2024.html"
    try:
        response = requests.get(url)
        tables = pd.read_html(response.text)
        df = tables[0]
        
        # Clean up the dataframe
        df.columns = df.columns.droplevel(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        df = df[df['Pk'].notna()]
        df['Pk'] = pd.to_numeric(df['Pk'], errors='coerce')
        df['Player'] = df['Player'].str.strip()
        
        return df[['Pk', 'Player', 'Tm']].dropna(subset=['Pk'])
    except Exception as e:
        st.error(f"Error fetching draft data: {e}")
        return None

@st.cache_data
def fetch_season_stats():
    """Scrape 2024-25 season stats from Basketball Reference"""
    url = "https://www.basketball-reference.com/leagues/NBA_2025.html"
    try:
        response = requests.get(url)
        tables = pd.read_html(response.text)
        df = tables[0]
        
        # Clean columns
        df.columns = df.columns.droplevel(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        df = df[df['Player'].notna()]
        df['Player'] = df['Player'].str.strip()
        
        # Convert stats to numeric
        numeric_cols = ['PTS', 'FG%', '3P%', 'TRB', 'AST', 'MP']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df[['Player', 'PTS', 'FG%', '3P%', 'TRB', 'AST', 'MP']].dropna(subset=['PTS'])
    except Exception as e:
        st.error(f"Error fetching season stats: {e}")
        return None

@st.cache_data
def calculate_ts_percent(df):
    """Calculate True Shooting percentage"""
    if 'FG' in df.columns and 'FTA' in df.columns and 'PTS' in df.columns:
        df['TS%'] = (df['PTS'] / (2 * (df['FG'] + 0.44 * df['FTA']))) * 100
    return df

# Fetch data
with st.spinner("Loading NBA data..."):
    draft_df = fetch_draft_data()
    season_df = fetch_season_stats()

if draft_df is not None and season_df is not None:
    # Merge datasets
    merged_df = draft_df.merge(season_df, on='Player', how='inner')
    merged_df = merged_df[merged_df['MP'] > 100]  # Filter for players with meaningful minutes
    
    if len(merged_df) > 0:
        # Sidebar controls
        st.sidebar.header("Customize Your Analysis")
        metric = st.sidebar.selectbox(
            "Select Performance Metric",
            ["PTS", "FG%", "3P%", "TRB", "AST"],
            help="Choose what stat to compare against draft position"
        )
        
        min_minutes = st.sidebar.slider(
            "Minimum Minutes Played",
            min_value=50,
            max_value=2000,
            value=100,
            step=100
        )
        
        # Filter by minutes
        filtered_df = merged_df[merged_df['MP'] > min_minutes].copy()
        
        # Calculate correlation
        correlation = filtered_df[['Pk', metric]].corr().iloc[0, 1]
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=filtered_df['Pk'],
            y=filtered_df[metric],
            mode='markers',
            marker=dict(
                size=filtered_df['MP'] / 30,
                color=filtered_df[metric],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=metric),
                line=dict(width=1, color='white')
            ),
            text=filtered_df['Player'],
            hovertemplate='<b>%{text}</b><br>Draft: %{x}<br>' + metric + ': %{y:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Draft Position vs {metric} (Correlation: {correlation:.2f})",
            xaxis_title="Draft Position (1 = Best)",
            yaxis_title=metric,
            hovermode='closest',
            height=600,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.header("📊 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Correlation Coefficient", f"{correlation:.2f}")
        
        with col2:
            st.metric("Players Analyzed", len(filtered_df))
        
        with col3:
            st.metric("Avg Minutes", f"{filtered_df['MP'].mean():.0f}")
        
        # Overperformers and underperformers
        st.subheader("Notable Performers")
        
        expected_performance = filtered_df['Pk'].mean()
        
        if metric in filtered_df.columns:
            filtered_df['Expected'] = expected_performance
            overperformers = filtered_df.nsmallest(5, 'Pk')[['Player', 'Pk', metric, 'MP']]
            underperformers = filtered_df.nlargest(5, 'Pk')[['Player', 'Pk', metric, 'MP']]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Best Draft Value (Late picks performing well)**")
                st.dataframe(overperformers, hide_index=True)
            
            with col2:
                st.write("**Early Picks Not Yet Delivering**")
                st.dataframe(underperformers, hide_index=True)
    else:
        st.error("No matching data found between draft and season stats")
else:
    st.error("Unable to load data. Please try refreshing the page.")