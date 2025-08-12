import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
import json

# --- 데이터 로드 (앱 실행 시 한 번만 수행) ---
print("데이터 로드 중...")
df_1m = pd.read_parquet('1m_analysis_results_5years_robust.parquet')
df_5m = pd.read_parquet('analysis_results_5years_robust.parquet')
print("데이터 로드 완료.")

# --- Dash 앱 생성 ---

app = dash.Dash(__name__)

app.layout = html.Div([
    # ❗️❗️❗️ 컨트롤 패널 전면 수정 ❗️❗️❗️
    html.Div([
        # 필터링 섹션
        html.Div([
            html.H4("필터링 조건"),
            html.Div([
                html.Label("피봇 수 범위:"),
                dcc.Input(id='pivot-min-input', type='number', value=4, style={'width': '60px', 'margin': '0 5px'}),
                dcc.Input(id='pivot-max-input', type='number', value=10, style={'width': '60px'}),
            ], style={'display': 'inline-block', 'padding': '10px'}),
            html.Div([
                html.Label("방향:"),
                dcc.Dropdown(
                    id='direction-dropdown',
                    options=[
                        {'label': '전체', 'value': 'all'},
                        {'label': 'UP', 'value': 'up'},
                        {'label': 'DOWN', 'value': 'down'}
                    ],
                    value='all',
                    clearable=False,
                    style={'width': '150px', 'display': 'inline-block', 'marginLeft': '10px', 'color': '#1E1E1E'}
                ),
            ], style={'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'middle'}),
        ], style={'padding': '10px', 'backgroundColor': '#2a2a2a', 'marginBottom': '10px'}),
        
        # 분석 파라미터 및 실행 섹션
        html.Div([
            html.H4("분석 파라미터"),
            html.Div([
                html.Label("UMAP Neighbors:", title="이웃 수. 낮으면 국소 구조, 높으면 전역 구조 강조."),
                dcc.Input(id='umap-neighbors-input', type='number', value=50, style={'width': '80px', 'marginLeft': '10px'}),
            ], style={'display': 'inline-block', 'padding': '10px'}),
            html.Div([
                html.Label("UMAP MinDist:", title="최소 거리. 낮으면 군집이 빽빽해지고, 높으면 느슨해짐."),
                dcc.Input(id='umap-mindist-input', type='number', value=0.5, step=0.1, style={'width': '80px', 'marginLeft': '10px'}),
            ], style={'display': 'inline-block', 'padding': '10px'}),
            html.Button('분석 실행', id='run-analysis-button', n_clicks=0, style={'padding': '8px 15px', 'marginLeft': '20px', 'verticalAlign': 'middle'}),
        ], style={'padding': '10px', 'backgroundColor': '#2a2a2a'}),
    ], style={'textAlign': 'center', 'padding': '10px'}),
    
    dcc.Loading(
        id="loading-component",
        children=[dcc.Graph(id='fractal-network-graph', style={'height': '80vh'})],
        type="default",
    ),

    dcc.Store(id='umap-results-store'),
    dcc.Store(id='clicked-point-store', data=None)
])

# --- 콜백 함수 ---

@app.callback(
    Output('umap-results-store', 'data'),
    Output('fractal-network-graph', 'figure'),
    Input('run-analysis-button', 'n_clicks'),
    # ❗️❗️❗️ 필터 값 State 추가 ❗️❗️❗️
    State('pivot-min-input', 'value'),
    State('pivot-max-input', 'value'),
    State('direction-dropdown', 'value'),
    State('umap-neighbors-input', 'value'),
    State('umap-mindist-input', 'value'),
    prevent_initial_call=True
)
def run_umap_analysis(n_clicks, p_min, p_max, direction, n_neighbors, min_dist):
    print(f"분석 시작: 피봇({p_min}~{p_max}), 방향({direction}), UMAP(n={n_neighbors}, d={min_dist})")

    # --- 필터링 로직 ---
    df_1m_filtered = df_1m[(df_1m['pivot_count'] >= p_min) & (df_1m['pivot_count'] <= p_max)]
    df_5m_filtered = df_5m[(df_5m['pivot_count'] >= p_min) & (df_5m['pivot_count'] <= p_max)]

    if direction == 'up':
        df_1m_filtered = df_1m_filtered[df_1m_filtered['direction'] == 1.0]
        df_5m_filtered = df_5m_filtered[df_5m_filtered['direction'] == 1.0]
    elif direction == 'down':
        df_1m_filtered = df_1m_filtered[df_1m_filtered['direction'] == -1.0]
        df_5m_filtered = df_5m_filtered[df_5m_filtered['direction'] == -1.0]
    
    print(f"필터링 결과: 1분봉 {len(df_1m_filtered)}개, 5분봉 {len(df_5m_filtered)}개")
    
    if df_1m_filtered.empty or df_5m_filtered.empty:
        return json.dumps({}), go.Figure().update_layout(title_text='필터 조건에 맞는 데이터가 없습니다.', template='plotly_dark')

    # --- UMAP 및 후처리 ---
    features_to_use = ['retracement_score', 'abs_angle_deg', 'pivot_count']
    combined_df = pd.concat([df_1m_filtered[features_to_use], df_5m_filtered[features_to_use]], ignore_index=True)
    scaled_features = StandardScaler().fit_transform(combined_df.values)

    reducer = umap.UMAP(n_neighbors=min(n_neighbors, len(scaled_features)-1), min_dist=min_dist, n_components=2, random_state=42)
    embedding = reducer.fit_transform(scaled_features)

    num_1m_patterns = len(df_1m_filtered)
    embedding_1m = embedding[:num_1m_patterns]
    embedding_5m = embedding[num_1m_patterns:]

    connections_by_parent = {}
    for i, parent in df_5m_filtered.iterrows():
        parent_start, parent_end = parent['start_ts'], parent['end_ts']
        children = df_1m_filtered[(df_1m_filtered['start_ts'] >= parent_start) & (df_1m_filtered['end_ts'] <= parent_end)]
        if not children.empty:
            connections_by_parent[i] = children.index.tolist()
    
    stored_data = {
        'embedding_1m': embedding_1m.tolist(),'embedding_5m': embedding_5m.tolist(),
        'df_1m_indices': df_1m_filtered.index.tolist(), 'df_5m_indices': df_5m_filtered.index.tolist(),
        'connections': connections_by_parent
    }
    
    base_fig = go.Figure()
    base_fig.add_trace(go.Scatter(
        x=embedding_1m[:, 0], y=embedding_1m[:, 1], mode='markers', name='1m Patterns (Child)',
        marker=dict(size=4, color='rgba(135, 206, 250, 0.7)'),
        hoverinfo='text', text=[f"1m - Index: {i}" for i in df_1m_filtered.index]
    ))
    base_fig.add_trace(go.Scatter(
        x=embedding_5m[:, 0], y=embedding_5m[:, 1], mode='markers', name='5m Patterns (Parent)',
        marker=dict(size=8, color='rgba(255, 165, 0, 0.6)', symbol='diamond'),
        customdata=df_5m_filtered.index,
        hoverinfo='text', text=[f"5m - Index: {i}" for i in df_5m_filtered.index]
    ))
    base_fig.update_layout(title=f"프랙탈 신경망 (필터링 적용)", template='plotly_dark', showlegend=True)

    print("분석 완료.")
    return json.dumps(stored_data), base_fig

# 클릭 콜백은 이전과 동일하게 작동 (수정 필요 없음)
@app.callback(
    Output('fractal-network-graph', 'figure', allow_duplicate=True),
    Output('clicked-point-store', 'data'),
    Input('fractal-network-graph', 'clickData'),
    State('umap-results-store', 'data'),
    State('clicked-point-store', 'data'),
    prevent_initial_call=True
)
def display_click_connections(clickData, stored_data_json, last_clicked_index):
    if stored_data_json is None or stored_data_json == '{}':
        return dash.no_update, dash.no_update
        
    stored_data = json.loads(stored_data_json)
    embedding_1m = np.array(stored_data['embedding_1m'])
    embedding_5m = np.array(stored_data['embedding_5m'])
    df_1m_indices = stored_data['df_1m_indices']
    df_5m_indices = stored_data['df_5m_indices']
    connections_by_parent_str_keys = stored_data['connections']
    connections_by_parent = {int(k): v for k, v in connections_by_parent_str_keys.items()}

    base_fig = go.Figure()
    base_fig.add_trace(go.Scatter(
        x=embedding_1m[:, 0], y=embedding_1m[:, 1], mode='markers', name='1m Patterns (Child)',
        marker=dict(size=4, color='rgba(135, 206, 250, 0.7)'),
        hoverinfo='text', text=[f"1m - Index: {i}" for i in df_1m_indices]
    ))
    base_fig.add_trace(go.Scatter(
        x=embedding_5m[:, 0], y=embedding_5m[:, 1], mode='markers', name='5m Patterns (Parent)',
        marker=dict(size=8, color='rgba(255, 165, 0, 0.6)', symbol='diamond'),
        customdata=df_5m_indices,
        hoverinfo='text', text=[f"5m - Index: {i}" for i in df_5m_indices]
    ))
    base_fig.update_layout(title=f"프랙탈 신경망 (필터링 적용)", template='plotly_dark', showlegend=True)

    if clickData is None or 'points' not in clickData:
        return base_fig, None

    if clickData['points'][0]['curveNumber'] != 1:
        return dash.no_update, last_clicked_index

    point_index = clickData['points'][0]['pointIndex']
    parent_original_idx = df_5m_indices[point_index]

    if parent_original_idx == last_clicked_index:
        return base_fig, None

    child_original_indices = connections_by_parent.get(parent_original_idx, [])
    
    if child_original_indices:
        map_5m_original_to_seq = {original_idx: seq_idx for seq_idx, original_idx in enumerate(df_5m_indices)}
        map_1m_original_to_seq = {original_idx: seq_idx for seq_idx, original_idx in enumerate(df_1m_indices)}

        parent_seq_idx = map_5m_original_to_seq.get(parent_original_idx)
        parent_coord = embedding_5m[parent_seq_idx]
        
        shapes = []
        for child_original_idx in child_original_indices:
            child_seq_idx = map_1m_original_to_seq.get(child_original_idx)
            if child_seq_idx is not None:
                child_coord = embedding_1m[child_seq_idx]
                shapes.append(go.layout.Shape(
                    type="line", x0=parent_coord[0], y0=parent_coord[1],
                    x1=child_coord[0], y1=child_coord[1],
                    line=dict(color="rgba(255, 255, 255, 0.5)", width=1)
                ))
        
        base_fig.update_layout(shapes=shapes)

    return base_fig, parent_original_idx

if __name__ == '__main__':
    app.run(debug=True, port=8051)