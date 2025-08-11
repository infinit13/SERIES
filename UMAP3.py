# professional_analyzer_dashboard_v4.4.py (Scaling Hotfix)

import dash
from dash import dcc, html
from dash import Input, Output, State, ctx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
import traceback

# --- 분석 라이브러리 임포트 ---
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
import umap # umap-learn >= 0.5 필요

# ==============================================================================
# ## 섹션 1: 데이터 로드 함수
# ==============================================================================
def fetch_klines(symbol: str, timeframe: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """지정된 기간의 K-line 데이터를 API 서버에서 가져옵니다."""
    url = "http://localhost:8202/api/klines"
    params = {"symbol": symbol.upper(), "interval": timeframe, "startTime": start_ts, "endTime": end_ts}
    print(f"   > Klines API 요청: {timeframe} ({start_ts} ~ {end_ts})")
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data:
            print("   > 경고: API에서 빈 데이터를 반환했습니다.")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        return df
    except requests.exceptions.RequestException as e:
        print(f"   > 데이터 가져오기 오류: {e}")
        return pd.DataFrame()

# ==============================================================================
# ## 섹션 2: Dash 앱 레이아웃 (자식 패널 추가)
# ==============================================================================
app = dash.Dash(__name__)
app.title = "Weighted Exploratory Pattern Analysis"

app.layout = html.Div(style={'backgroundColor': '#1E1E1E', 'color': '#E0E0E0', 'fontFamily': 'sans-serif'}, children=[
    dcc.Store(id='analysis-data-store'),
    html.H1(children='가중 UMAP 탐색적 패턴 분석 플랫폼', style={'textAlign': 'center', 'padding': '15px'}),
    
    html.Div([
        html.Div([
            html.H4("1. 데이터 로드", style={'marginTop': '0', 'marginBottom': '10px'}),
            dcc.Input(id='parent-filepath-input', type='text', placeholder='부모 TF 파일 경로 (예: 15분봉)', value='15m_analysis_results_5years_robust.parquet', style={'width': '300px', 'marginRight':'10px'}),
            dcc.Input(id='child-filepath-input', type='text', placeholder='자식 TF 파일 경로 (예: 5분봉)', value='analysis_results_5years_robust.parquet', style={'width': '300px'}),
            html.Button('데이터 불러오기', id='load-data-button', n_clicks=0, style={'padding': '8px 15px', 'marginLeft': '10px'}),
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px', 'marginBottom': '10px'}),

        html.Div([
            html.H4("2. 데이터 필터링", style={'marginTop': '0', 'marginBottom': '10px'}),
            html.Div([
                html.Div([html.Label("되돌림 점수:"), dcc.Input(id='ret-score-min', type='number', placeholder='최소'), dcc.Input(id='ret-score-max', type='number', placeholder='최대')], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([html.Label("피봇 개수:"), dcc.Input(id='pivot-min', type='number', placeholder='최소'), dcc.Input(id='pivot-max', type='number', placeholder='최대')], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([html.Label("절대 각도:"), dcc.Input(id='slope-min', type='number', placeholder='최소'), dcc.Input(id='slope-max', type='number', placeholder='최대')], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([html.Label("방향:"), dcc.Dropdown(id='direction-dropdown', options=[{'label': '전체', 'value': 'all'}, {'label': 'UP', 'value': 'up'}, {'label': 'DOWN', 'value': 'down'}], value='all', clearable=False)], style={'display': 'inline-block', 'width': '150px', 'padding': '5px 10px', 'verticalAlign': 'middle'}),
            ], style={'textAlign': 'center'}),
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px', 'marginBottom': '10px'}),

        html.Div([
            html.H4("3. 분석 및 시각화 설정", style={'marginTop': '0', 'marginBottom': '10px'}),
            html.Div([
                html.Div([html.Label("UMAP Neighbors:"), dcc.Input(id='umap-neighbors-input', type='number', value=15, style={'width': '60px', 'marginLeft': '5px'})], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([html.Label("UMAP MinDist:"), dcc.Input(id='umap-mindist-input', type='number', value=0.1, step=0.05, style={'width': '60px', 'marginLeft': '5px'})], style={'display': 'inline-block', 'padding': '5px 10px'}),
            ], style={'textAlign': 'center'}),
            
            html.Div([
                html.Label("Gamma (자식 세력 강조 강도):", title="이 값을 높이면 UMAP이 자식의 세력 패턴을 더 강하게 반영합니다."),
                dcc.Slider(id='gamma-slider', min=0, max=1, step=0.05, value=0.25, marks={i/10:str(i/10) for i in range(11)})
            ], style={'width': '50%', 'margin': 'auto', 'padding': '10px'}),

            html.Div([
                dcc.Dropdown(
                    id='color-selector',
                    options=[
                        {'label': '채색: 자식 세력 등급', 'value': 'child_strength'},
                        {'label': '채색: 자식 총점수', 'value': 'child_sum_force'},
                        {'label': '채색: 부모 세력 점수', 'value': 'force_score'},
                        {'label': '채색: 방향 (UP/DOWN)', 'value': 'direction'},
                    ],
                    value='child_strength', clearable=False,
                    style={'width': '250px', 'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '10px'}
                ),
                ### 수정된 부분 ###
                # 자식 총점수 채색 시 사용할 스케일링 방법 선택 UI
                dcc.RadioItems(
                    id='scaling-method-selector',
                    options=[
                        {'label': '원시값', 'value': 'raw'},
                        {'label': '정규화', 'value': 'normalized'},
                        {'label': '로그 변환', 'value': 'log'},
                    ],
                    value='raw',
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                    style={'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '20px'}
                ),
                html.Button('분석 및 시각화 실행', id='run-button', n_clicks=0, style={'padding': '8px 15px'}),
            ], style={'textAlign': 'center', 'marginTop': '10px'}),
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px'}),
    ], style={'textAlign': 'center', 'padding': '10px'}),

    # 메인 차트와 자식 패널 레이아웃
    dcc.Loading(id="loading-icon", children=[dcc.Graph(id='main-chart', style={'height': '60vh'})], type="default"),
    dcc.Loading(dcc.Graph(id='child-overview', style={'height': '35vh'})),
    html.Div(id='click-output', style={'textAlign': 'center', 'padding': '10px', 'fontSize': '16px', 'fontWeight': 'bold'}),
])

# ==============================================================================
# ## 섹션 3: Dash 콜백
# ==============================================================================

def get_force_score(df):
    """세력 점수를 계산하는 헬퍼 함수"""
    return df['retracement_score'] * df['abs_angle_deg'].abs()

@app.callback(
    Output('analysis-data-store', 'data'),
    Input('load-data-button', 'n_clicks'),
    State('parent-filepath-input', 'value'),
    State('child-filepath-input', 'value'),
    prevent_initial_call=True
)
def load_analysis_data(n_clicks, parent_path, child_path):
    if not parent_path or not child_path: return None
    try:
        df_parent = pd.read_parquet(parent_path)
        df_child = pd.read_parquet(child_path)

        df_parent['force_score'] = get_force_score(df_parent)
        df_child['force_score'] = get_force_score(df_child)

        child_max_force = []
        child_sum_force = []
        for parent_row in df_parent.itertuples():
            children_in_range = df_child[
                (df_child['start_ts'] >= parent_row.start_ts) & 
                (df_child['end_ts'] <= parent_row.end_ts)
            ]
            child_max_force.append(children_in_range['force_score'].max() if not children_in_range.empty else 0)
            child_sum_force.append(children_in_range['force_score'].sum() if not children_in_range.empty else 0)
        
        df_parent['child_max_force'] = child_max_force
        df_parent['child_sum_force'] = child_sum_force
        
        strong_children_mask = df_parent['child_max_force'] > 0
        y = pd.Series(-1, index=df_parent.index, dtype=int)
        
        if strong_children_mask.any():
            qt = QuantileTransformer(n_quantiles=5, output_distribution='uniform', random_state=42)
            scores_to_transform = df_parent.loc[strong_children_mask, ['child_max_force']]
            if len(scores_to_transform) > 0 and scores_to_transform.nunique().iloc[0] < 5:
                 unique_count = scores_to_transform.nunique().iloc[0]
                 qt.n_quantiles = unique_count if unique_count > 0 else 1
            if qt.n_quantiles > 1:
                y[strong_children_mask] = pd.cut(qt.fit_transform(scores_to_transform).flatten(), bins=qt.n_quantiles, labels=False, include_lowest=True)
            else:
                 y[strong_children_mask] = 0
        df_parent['y_child_strength_label'] = y
        return {'parent_data': df_parent.to_dict('records'), 'child_data_full': df_child.to_dict('records')}
    except Exception as e:
        traceback.print_exc()
        return None

@app.callback(
    Output('main-chart', 'figure'),
    Output('child-overview', 'figure'),
    Output('click-output', 'children'),
    Input('run-button', 'n_clicks'),
    Input('main-chart', 'clickData'),
    State('analysis-data-store', 'data'),
    State('ret-score-min', 'value'), State('ret-score-max', 'value'),
    State('pivot-min', 'value'), State('pivot-max', 'value'),
    State('slope-min', 'value'), State('slope-max', 'value'),
    State('direction-dropdown', 'value'),
    State('color-selector', 'value'),
    State('umap-neighbors-input', 'value'),
    State('umap-mindist-input', 'value'),
    State('gamma-slider', 'value'),
    ### 수정된 부분 ###
    # 스케일링 방법 선택 값(state) 추가
    State('scaling-method-selector', 'value'),
    prevent_initial_call=True
)
def universal_callback(
    run_clicks, clickData, analysis_data,
    rs_min, rs_max, p_min, p_max, s_min, s_max, direction,
    color_mode, umap_neighbors, umap_min_dist, gamma,
    scaling_method # 스케일링 방법 값
):
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'
    df_parent_full = pd.DataFrame(analysis_data['parent_data'])

    if triggered_id == 'main-chart' and clickData:
        df_child_full = pd.DataFrame(analysis_data['child_data_full'])
        if 'customdata' not in clickData['points'][0]:
            msg = "오류: 클릭 데이터에 'customdata'가 없습니다."
            return dash.no_update, go.Figure().update_layout(template='plotly_dark', title=msg), msg
        pid = clickData['points'][0]['customdata']
        P = df_parent_full.loc[pid]
        kids = df_child_full[(df_child_full['start_ts'] >= P['start_ts']) & (df_child_full['end_ts'] <= P['end_ts'])].copy()
        if kids.empty:
            msg = f"부모 #{pid}: 자식 없음"
            return dash.no_update, go.Figure().update_layout(template='plotly_dark', title="자식 없음"), msg

        kids['force_score'] = get_force_score(kids)
        kids = kids.sort_values('start_ts').reset_index(drop=True)
        kids['rank'] = kids.index + 1
        kids['duration'] = (kids['end_ts'] - kids['start_ts']) / 1000
        
        ### 수정된 부분 ###
        # 자식 패널의 세력점수 색상을 로컬 0-1 스케일로 조정
        force_scores = kids['force_score'].values
        min_force, max_force = force_scores.min(), force_scores.max()
        local_scaled_color = (force_scores - min_force) / (max_force - min_force) if max_force > min_force else np.full(len(force_scores), 0.5)

        hover = "Rank: %{y}<br>Force: %{customdata[2]:.2f}<br>Dur(s): %{x:.0f}<br>start: %{customdata[0]}<br>end: %{customdata[1]}"
        fig_child = go.Figure(go.Bar(
            y=kids['rank'], x=kids['duration'], orientation='h',
            marker=dict(color=local_scaled_color, colorscale='Plasma', showscale=True, colorbar=dict(title='Local Force (0-1)')),
            customdata=np.c_[pd.to_datetime(kids['start_ts'], unit='ms').astype(str), pd.to_datetime(kids['end_ts'], unit='ms').astype(str), kids['force_score']],
            hovertemplate=hover
        ))
        fig_child.update_layout(template='plotly_dark', title=f"자식 일괄 보기 · 부모 #{pid} (자식 {len(kids)}개)",
                                xaxis_title="지속시간(초)", yaxis_title="자식 순번(시간순)", yaxis=dict(autorange="reversed"), margin=dict(l=40, r=20, b=40, t=40))
        msg = f"부모 #{pid}: 자식 {len(kids)}개, 자식 총점수: {kids['force_score'].sum():.2f}"
        return dash.no_update, fig_child, msg

    elif triggered_id == 'run-button':
        query_parts = []
        if rs_min is not None: query_parts.append(f"retracement_score >= {rs_min}")
        if rs_max is not None: query_parts.append(f"retracement_score <= {rs_max}")
        if p_min is not None: query_parts.append(f"pivot_count >= {p_min}")
        if p_max is not None: query_parts.append(f"pivot_count <= {p_max}")
        if s_min is not None: query_parts.append(f"abs_angle_deg >= {s_min}")
        if s_max is not None: query_parts.append(f"abs_angle_deg <= {s_max}")
        if direction != 'all': query_parts.append(f"direction == {1.0 if direction == 'up' else -1.0}")
        query_str = " and ".join(query_parts)
        df_filtered = df_parent_full.query(query_str) if query_parts else df_parent_full
        
        if len(df_filtered) < 2:
            fig = go.Figure().update_layout(title_text='필터 조건에 맞는 데이터가 너무 적습니다.', template='plotly_dark')
            return fig, go.Figure().update_layout(template='plotly_dark', title=""), "데이터 부족"

        features = df_filtered[['retracement_score', 'abs_angle_deg', 'pivot_count']].values
        scaled_features = StandardScaler().fit_transform(features)
        y_supervised = df_filtered['y_child_strength_label'].values
        reducer = umap.UMAP(n_neighbors=umap_neighbors, min_dist=umap_min_dist, random_state=42, target_weight=gamma)
        embedding = reducer.fit_transform(scaled_features, y=y_supervised)
        
        hover_texts = [f"인덱스: {idx}<br>부모점수: {r['force_score']:.2f}<br>자식최대: {r['child_max_force']:.2f}<br><b>자식총점: {r['child_sum_force']:.2f}</b><br>자식등급: {r['y_child_strength_label']}" for idx, r in df_filtered.iterrows()]
        
        marker_color, colorscale, cbar_title = df_filtered['force_score'], 'Viridis', '부모 세력 점수'
        
        if color_mode == 'child_strength':
            marker_color, colorscale, cbar_title = df_filtered['y_child_strength_label'], 'Plasma', '자식 세력 등급'
        elif color_mode == 'direction':
            marker_color, colorscale, cbar_title = df_filtered['direction'].map({1.0: 1, -1.0: 0}), [[0, 'lightcoral'], [1, 'lightgreen']], '방향'
        ### 수정된 부분 ###
        # '자식 총점수' 채색 시 스케일링 로직 적용
        elif color_mode == 'child_sum_force':
            colorscale, cbar_title_base = 'Cividis', '자식 총점수'
            scores = df_filtered['child_sum_force'].values.reshape(-1, 1)
            
            if scaling_method == 'normalized':
                marker_color = MinMaxScaler().fit_transform(scores)
                cbar_title = f'{cbar_title_base} (정규화)'
            elif scaling_method == 'log':
                marker_color = np.log1p(scores)
                cbar_title = f'{cbar_title_base} (로그 변환)'
            else: # 'raw'
                marker_color = scores
                cbar_title = cbar_title_base

        fig = go.Figure(data=[go.Scattergl(x=embedding[:, 0], y=embedding[:, 1], mode='markers', customdata=df_filtered.index.to_list(),
                                          marker=dict(size=7, color=marker_color, colorscale=colorscale, showscale=True, opacity=0.8, colorbar={'title': cbar_title}),
                                          text=hover_texts, hoverinfo='text')])
        fig.update_layout(template='plotly_dark', title_text=f'2D UMAP 사영 | 데이터: {len(df_filtered)}개, Gamma: {gamma}', xaxis_title='UMAP 1', yaxis_title='UMAP 2', margin=dict(l=20, r=20, b=20, t=60))
        return fig, go.Figure().update_layout(template='plotly_dark', title="부모 클릭 시 자식 표시"), f"필터링된 데이터: {len(df_filtered)}개"

    return dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    print("\n### 가중 UMAP 탐색적 패턴 분석 플랫폼 v4.4 (스케일링 기능 추가) ###")
    app.run(debug=True, port=8052)