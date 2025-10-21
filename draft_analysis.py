import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from io import StringIO
import numpy as np

st.set_page_config(page_title="NBA Draft vs Performance", layout="wide")
st.title("NBA Draft Position vs Current Season Performance")
st.markdown("Analysing how draft position correlates with 2024-25 season performance")

@st.cache_data
def fetch_draft_data():
    """Scrape 2024 NBA draft data from Basketball Reference"""
    url = "https://www.basketball-reference.com/draft/NBA_2024.html"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        
        pk_col = 'Pk' if 'Pk' in df.columns else df.columns[0]
        player_col = 'Player' if 'Player' in df.columns else df.columns[1]
        tm_col = 'Tm' if 'Tm' in df.columns else df.columns[2]
        
        df = df[[pk_col, player_col, tm_col]].copy()
        df.columns = ['Pk', 'Player', 'Tm']
        
        df = df[df['Pk'].notna()]
        df['Pk'] = pd.to_numeric(df['Pk'], errors='coerce')
        df['Player'] = df['Player'].astype(str).str.strip()
        df['Player_Clean'] = df['Player'].str.replace(r'\s+(Jr\.|Sr\.|III|II|IV)\.?$', '', regex=True).str.strip()
        
        return df.dropna(subset=['Pk'])
    except Exception as e:
        st.error(f"Error fetching draft data: {str(e)}")
        return None

@st.cache_data
def fetch_season_stats():
    """Scrape 2024-25 season stats from Basketball Reference"""
    url = "https://www.basketball-reference.com/leagues/NBA_2025_per_game.html"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        
        df = df[df['Player'].notna()].copy()
        df['Player'] = df['Player'].astype(str).str.strip()
        df['Player_Clean'] = df['Player'].str.replace(r'\s+(Jr\.|Sr\.|III|II|IV)\.?$', '', regex=True).str.strip()
        
        numeric_cols = ['PTS', 'FG%', '3P%', 'TRB', 'AST', 'STL', 'BLK', 'MP']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Composite Score'] = (
            df['PTS'] * 1.0 + 
            df['AST'] * 1.5 + 
            df['TRB'] * 1.2 + 
            df['STL'] * 3.0 + 
            df['BLK'] * 3.0
        ).round(2)
        
        return df[['Player', 'Player_Clean', 'PTS', 'FG%', '3P%', 'TRB', 'AST', 'STL', 'BLK', 'MP', 'Composite Score']].dropna(subset=['PTS'])
    except Exception as e:
        st.error(f"Error fetching season stats: {str(e)}")
        return None

with st.spinner("Loading NBA data..."):
    draft_df = fetch_draft_data()
    season_df = fetch_season_stats()

if draft_df is not None and season_df is not None:
    merged_df = draft_df.merge(season_df, left_on='Player_Clean', right_on='Player_Clean', how='inner', suffixes=('_draft', '_stats'))
    merged_df['Player'] = merged_df['Player_draft']
    # Use drafted team, not current team
    merged_df['Drafted_Team'] = merged_df['Tm']
    merged_df['Tm'] = merged_df['Drafted_Team'].fillna('N/A')
    
    if len(merged_df) > 0:
        tab1, tab2, tab3 = st.tabs(["Interactive Dashboard", "Player Comparison", "Team View"])
        
        with tab1:
            st.sidebar.header("Analysis Filters")
            
            search_term = st.sidebar.text_input("Search for a player", "", help="Type any part of a player's name")
            
            metric = st.sidebar.selectbox(
                "Select Performance Metric",
                ["Composite Score", "PTS", "AST", "TRB", "STL", "BLK", "FG%", "3P%"],
                help="Composite Score combines all stats for overall performance"
            )
            
            min_minutes = st.sidebar.slider(
                "Minimum Minutes Played",
                min_value=0,
                max_value=int(merged_df['MP'].max()),
                value=15,
                step=5
            )
            
            filtered_df = merged_df[merged_df['MP'] >= min_minutes].copy()
            
            if search_term:
                search_lower = search_term.lower().strip()
                filtered_df = filtered_df[
                    filtered_df['Player'].str.lower().str.contains(search_lower, na=False, regex=False)
                ]
                if len(filtered_df) == 0:
                    st.warning(f"No players found matching '{search_term}'. Try a different spelling or fewer characters.")
                else:
                    st.info(f"Found {len(filtered_df)} player(s) matching '{search_term}'")
            
            if len(filtered_df) >= 1:
                if len(filtered_df) > 1:
                    correlation = filtered_df[['Pk', metric]].corr().iloc[0, 1]
                else:
                    correlation = 0
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=filtered_df['Pk'],
                    y=filtered_df[metric],
                    mode='markers',
                    marker=dict(
                        size=filtered_df['MP'] / 15,
                        color=filtered_df[metric],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=metric),
                        line=dict(width=1, color='white'),
                        sizemin=8
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
                
                st.header("📊 Key Insights")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Correlation Coefficient", f"{correlation:.2f}")
                    if abs(correlation) > 0.5:
                        st.caption("Strong relationship")
                    elif abs(correlation) > 0.3:
                        st.caption("Moderate relationship")
                    else:
                        st.caption("Weak relationship")
                
                with col2:
                    st.metric("Players Analyzed", len(filtered_df))
                
                with col3:
                    st.metric("Avg Minutes", f"{filtered_df['MP'].mean():.0f}")
                
                st.subheader("Notable Performers")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top 5 Picks Performance**")
                    top_picks = filtered_df[filtered_df['Pk'] <= 5][['Player', 'Pk', metric, 'MP']].sort_values('Pk')
                    if len(top_picks) > 0:
                        st.dataframe(top_picks, hide_index=True)
                    else:
                        st.info("Not enough top picks with sufficient minutes")
                
                with col2:
                    st.write("**Best Late Draft Finds (Picks 20+)**")
                    late_picks = filtered_df[filtered_df['Pk'] >= 20].nlargest(5, metric)[['Player', 'Pk', metric, 'MP']]
                    if len(late_picks) > 0:
                        st.dataframe(late_picks, hide_index=True)
                    else:
                        st.info("Not enough late picks with sufficient minutes")
                
                with st.expander("View Full Dataset"):
                    display_cols = ['Pk', 'Player', 'Tm', 'Composite Score', 'PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P%', 'MP']
                    display_df = filtered_df[display_cols].sort_values('Pk').copy()
                    display_df.rename(columns={'Tm': 'Drafted By'}, inplace=True)
                    st.dataframe(display_df, hide_index=True)
                    
                if metric == "Composite Score":
                    with st.expander("ℹ️ How is Composite Score calculated?"):
                        st.markdown("""
                        **Composite Score** combines multiple stats into one overall performance metric:
                        - Points × 1.0
                        - Assists × 1.5 (weighted higher for playmaking)
                        - Rebounds × 1.2 (weighted for two-way impact)
                        - Steals × 3.0 (weighted heavily for defensive impact)
                        - Blocks × 3.0 (weighted heavily for rim protection)
                        
                        Higher scores indicate better all-around performance.
                        """)
            else:
                st.warning("Not enough players meet the minimum minutes threshold. Try lowering it.")
        
        with tab2:
            st.subheader("⚖️ Compare Two Players")
            
            col1, col2 = st.columns(2)
            
            player_list = sorted(merged_df['Player'].unique().tolist())
            
            with col1:
                player1 = st.selectbox("Select Player 1", player_list, key="player1")
            
            with col2:
                player2 = st.selectbox("Select Player 2", player_list, key="player2", index=min(1, len(player_list)-1))
            
            if player1 and player2:
                p1_data = merged_df[merged_df['Player'] == player1].iloc[0]
                p2_data = merged_df[merged_df['Player'] == player2].iloc[0]
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### {player1}")
                    st.metric("Draft Position", f"#{int(p1_data['Pk'])}")
                    st.metric("Drafted By", p1_data['Tm'])
                    st.metric("Minutes Played", f"{p1_data['MP']:.1f}")
                
                with col2:
                    st.markdown(f"### {player2}")
                    st.metric("Draft Position", f"#{int(p2_data['Pk'])}")
                    st.metric("Drafted By", p2_data['Tm'])
                    st.metric("Minutes Played", f"{p2_data['MP']:.1f}")
                
                st.markdown("---")
                st.markdown("#### Statistical Comparison")
                
                stats_to_compare = ['Composite Score', 'PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P%']
                
                comparison_data = []
                for stat in stats_to_compare:
                    if stat in p1_data and stat in p2_data:
                        comparison_data.append({
                            'Stat': stat,
                            player1: round(p1_data[stat], 2) if pd.notna(p1_data[stat]) else 0,
                            player2: round(p2_data[stat], 2) if pd.notna(p2_data[stat]) else 0
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, hide_index=True, use_container_width=True)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name=player1,
                    x=[row['Stat'] for row in comparison_data],
                    y=[row[player1] for row in comparison_data],
                    marker_color='#636EFA'
                ))
                
                fig.add_trace(go.Bar(
                    name=player2,
                    x=[row['Stat'] for row in comparison_data],
                    y=[row[player2] for row in comparison_data],
                    marker_color='#EF553B'
                ))
                
                fig.update_layout(
                    title=f"{player1} vs {player2}",
                    barmode='group',
                    xaxis_title="Statistic",
                    yaxis_title="Value",
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("🏀 Team View")
            
            teams = sorted([t for t in merged_df['Tm'].unique() if t and t != 'N/A'])
            selected_team = st.selectbox("Select a Team", teams)
            
            if selected_team:
                team_df = merged_df[merged_df['Tm'] == selected_team].sort_values('Pk')
                
                st.markdown(f"### {selected_team} - 2024 Draft Class")
                st.metric("Total Players Drafted", len(team_df))
                
                if len(team_df) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Avg Draft Position", f"{team_df['Pk'].mean():.1f}")
                    with col2:
                        st.metric("Total Minutes", f"{team_df['MP'].sum():.0f}")
                    with col3:
                        st.metric("Avg Composite Score", f"{team_df['Composite Score'].mean():.1f}")
                    with col4:
                        st.metric("Avg Points", f"{team_df['PTS'].mean():.1f}")
                    
                    st.markdown("---")
                    st.markdown("#### Players")
                    
                    display_cols = ['Pk', 'Player', 'Composite Score', 'PTS', 'AST', 'TRB', 'STL', 'BLK', 'MP']
                    st.dataframe(team_df[display_cols], hide_index=True, use_container_width=True)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=team_df['Pk'],
                        y=team_df['Composite Score'],
                        mode='markers+text',
                        marker=dict(
                            size=20,
                            color=team_df['MP'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Minutes"),
                            line=dict(width=1, color='white')
                        ),
                        text=team_df['Player'],
                        textposition="top center",
                        hovertemplate='<b>%{text}</b><br>Draft: %{x}<br>Composite: %{y:.1f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_team} Draft Picks - Performance vs Position",
                        xaxis_title="Draft Position",
                        yaxis_title="Composite Score",
                        height=500,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No players found for this team")
    else:
        st.error("No matching data found between draft and season stats")
else:
    st.error("Unable to load data. Please try refreshing the page.")