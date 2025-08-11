import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler

# --- 데이터 로드 및 전처리 (앱 실행 시 한 번만 수행) ---

print("데이터 로드 및 UMAP 계산을 시작합니다. 시간이 걸릴 수 있습니다...")

# 1. 데이터 소스 확보
df_1m = pd.read_parquet('1m_analysis_results_5years_robust.parquet')
df_5m = pd.read_parquet('analysis_results_5years_robust.parquet')
print(f"로드 완료: 1분봉 {len(df_1m)}개, 5분봉 {len(df_5m)}개")

# ❗️❗️❗️ 피봇 필터링 로직 추가 ❗️❗️❗️
min_pivots = 4
max_pivots = 10
df_1m = df_1m[(df_1m['pivot_count'] >= min_pivots) & (df_1m['pivot_count'] <= max_pivots)]
df_5m = df_5m[(df_5m['pivot_count'] >= min_pivots) & (df_5m['pivot_count'] <= max_pivots)]
print(f"필터링 완료 ({min_pivots}~{max_pivots}피봇): 1분봉 {len(df_1m)}개, 5분봉 {len(df_5m)}개")
# ❗️❗️❗️ 여기까지 ❗️❗️❗️

# 2. 데이터 통합 및 UMAP 계산
features_to_use = ['retracement_score', 'abs_angle_deg', 'pivot_count']
combined_df = pd.concat([df_1m[features_to_use], df_5m[features_to_use]], ignore_index=True)
scaled_features = StandardScaler().fit_transform(combined_df.values)

reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=2, random_state=42)
embedding = reducer.fit_transform(scaled_features)

num_1m_patterns = len(df_1m)
embedding_1m = embedding[:num_1m_patterns]
embedding_5m = embedding[num_1m_patterns:]

# 3. 부모-자식 관계 미리 계산
connections_by_parent = {}
for i, parent in df_5m.iterrows():
    parent_start, parent_end = parent['start_ts'], parent['end_ts']
    children = df_1m[(df_1m['start_ts'] >= parent_start) & (df_1m['end_ts'] <= parent_end)]
    if not children.empty:
        connections_by_parent[i] = children.index.tolist()

print("사전 계산 완료. Dash 앱을 시작합니다.")

# --- Dash 앱 생성 ---

app = dash.Dash(__name__)

# 기본 Figure 생성 (점들만 포함)
base_fig = go.Figure()
base_fig.add_trace(go.Scatter(
    x=embedding_1m[:, 0], y=embedding_1m[:, 1], mode='markers', name='1m Patterns (Child)',
    marker=dict(size=4, color='rgba(135, 206, 250, 0.7)'),
    hoverinfo='text',
    text=[f"1m - Index: {i}" for i in df_1m.index]
))
base_fig.add_trace(go.Scatter(
    x=embedding_5m[:, 0], y=embedding_5m[:, 1], mode='markers', name='5m Patterns (Parent)',
    marker=dict(size=8, color='rgba(255, 165, 0, 0.6)', symbol='diamond'),
    customdata=df_5m.index,
    hoverinfo='text', text=[f"5m - Index: {i}" for i in df_5m.index]
))
base_fig.update_layout(title="'클릭 토글 프랙탈 신경망' - 5분봉 패턴을 클릭하여 연결망 확인",
                       template='plotly_dark', showlegend=True)

app.layout = html.Div([
    dcc.Graph(id='fractal-network-graph', figure=base_fig, style={'height': '95vh'}),
    dcc.Store(id='clicked-point-store', data=None)
])

# --- 콜백 함수 (클릭 토글 로직) ---
@app.callback(
    Output('fractal-network-graph', 'figure'),
    Output('clicked-point-store', 'data'),
    Input('fractal-network-graph', 'clickData'),
    State('clicked-point-store', 'data'),
    prevent_initial_call=True
)
def display_click_connections(clickData, last_clicked_index):
    fig = go.Figure(data=base_fig.data, layout=base_fig.layout)
    
    if clickData is None or 'points' not in clickData:
        return base_fig, None

    if clickData['points'][0]['curveNumber'] != 1:
        return dash.no_update, last_clicked_index

    point_index = clickData['points'][0]['pointIndex']
    parent_original_idx = df_5m.index[point_index]

    if parent_original_idx == last_clicked_index:
        return base_fig, None

    child_indices = connections_by_parent.get(parent_original_idx, [])
    
    if child_indices:
        map_5m_original_to_seq = {original_idx: seq_idx for seq_idx, original_idx in enumerate(df_5m.index)}
        map_1m_original_to_seq = {original_idx: seq_idx for seq_idx, original_idx in enumerate(df_1m.index)}

        parent_seq_idx = map_5m_original_to_seq.get(parent_original_idx)
        parent_coord = embedding_5m[parent_seq_idx]
        
        shapes = []
        for child_original_idx in child_indices:
            child_seq_idx = map_1m_original_to_seq.get(child_original_idx)
            if child_seq_idx is not None:
                child_coord = embedding_1m[child_seq_idx]
                shapes.append(go.layout.Shape(
                    type="line",
                    x0=parent_coord[0], y0=parent_coord[1],
                    x1=child_coord[0], y1=child_coord[1],
                    line=dict(color="rgba(255, 255, 255, 0.5)", width=1)
                ))
        
        fig.update_layout(shapes=shapes)

    return fig, parent_original_idx

if __name__ == '__main__':
    app.run(debug=True, port=8051)