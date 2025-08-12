# professional_analyzer_dashboard_v5.2.py (Parent-Info-Enriched Child UMAP)

import dash
from dash import dcc, html
from dash import Input, Output, State, ctx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import traceback
import requests

# --- 분석 라이브러리 임포트 ---
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import umap # umap-learn >= 0.5 필요

# ==============================================================================
# ## 섹션 0: 서버 연동 설정
# ==============================================================================
SERVER_URL = "http://localhost:8202"
SYMBOL = "BTCUSDT"
HI_TF = "1h"

# ==============================================================================
# ## 섹션 1: 데이터 로드 및 서버 조회 함수
# ==============================================================================

def fetch_klines(server_url, symbol, timeframe, start_ms, end_ms):
    params = {"symbol": symbol, "timeframe": timeframe, "startTime": int(start_ms), "endTime": int(end_ms)}
    try:
        r = requests.get(f"{server_url}/api/klines", params=params, timeout=10)
        r.raise_for_status()
        js = r.json()
        if not js: return pd.DataFrame()
        df = pd.DataFrame(js, columns=['start_ts', 'open', 'high', 'low', 'close', 'volume'])
        df["ts"] = pd.to_datetime(df["start_ts"], unit="ms")
        df.set_index("ts", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
    except requests.exceptions.RequestException as e:
        print(f"서버로부터 캔들 데이터를 가져오는 데 실패했습니다: {e}")
        return pd.DataFrame()

def get_force_score(df):
    return df['retracement_score'] * df['abs_angle_deg'].abs()

# [핵심] 자식 노드에 부모 정보를 미리 연결해주는 데이터 보강 로직 추가
def load_analysis_data(parent_path, child_path):
    try:
        df_parent = pd.read_parquet(parent_path)
        df_child_full = pd.read_parquet(child_path)

        df_parent['force_score'] = get_force_score(df_parent)
        df_child_full['force_score'] = get_force_score(df_child_full)

        # 자식 데이터프레임에 부모 정보를 담을 컬럼 초기화
        df_child_full['parent_id'] = -1
        df_child_full['parent_force_score'] = np.nan

        # 각 부모에 대해 자식들을 매핑
        for parent_id, parent_row in df_parent.iterrows():
            mask = (df_child_full['start_ts'] >= parent_row['start_ts']) & \
                   (df_child_full['end_ts'] <= parent_row['end_ts'])
            
            df_child_full.loc[mask, 'parent_id'] = parent_id
            df_child_full.loc[mask, 'parent_force_score'] = parent_row['force_score']

        return {
            'parent_data': df_parent.to_dict('records'), 
            'child_data_full': df_child_full.to_dict('records') # 부모 정보가 포함된 자식 데이터
        }
    except Exception as e:
        traceback.print_exc()
        return None

# ==============================================================================
# ## 섹션 2: Dash 앱 레이아웃 (변경 없음)
# ==============================================================================
app = dash.Dash(__name__)
app.title = "Child-Node-Centric UMAP Analysis"

analysis_data_store = load_analysis_data(
    '1h_analysis_results_5years_robust.parquet',
    '15m_analysis_results_5years_robust.parquet'
)

app.layout = html.Div(style={'backgroundColor': '#1E1E1E', 'color': '#E0E0E0', 'fontFamily': 'sans-serif'}, children=[
    dcc.Store(id='analysis-data-store', data=analysis_data_store),
    html.H1(children='UMAP 자식 노드 패턴 분석', style={'textAlign': 'center', 'padding': '15px'}),
    
    html.Div([
        html.Div([
            html.H4("1. 자식 노드 필터링", style={'marginTop': '0', 'marginBottom': '10px'}),
            html.Div([
                html.Div([html.Label("되돌림 점수:"), dcc.Input(id='ret-score-min', type='number', placeholder='최소'), dcc.Input(id='ret-score-max', type='number', placeholder='최대')], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([html.Label("피봇 개수:"), dcc.Input(id='pivot-min', type='number', placeholder='최소'), dcc.Input(id='pivot-max', type='number', placeholder='최대')], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([html.Label("절대 각도:"), dcc.Input(id='slope-min', type='number', placeholder='최소'), dcc.Input(id='slope-max', type='number', placeholder='최대')], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([html.Label("방향:"), dcc.Dropdown(id='direction-dropdown', options=[{'label': '전체', 'value': 'all'}, {'label': 'UP', 'value': 'up'}, {'label': 'DOWN', 'value': 'down'}], value='all', clearable=False)], style={'display': 'inline-block', 'width': '150px', 'padding': '5px 10px', 'verticalAlign': 'middle'}),
            ], style={'textAlign': 'center'}),
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px', 'marginBottom': '10px'}),

        html.Div([
            html.H4("2. 분석 및 시각화 설정", style={'marginTop': '0', 'marginBottom': '10px'}),
            html.Div([
                html.Div([html.Label("UMAP Neighbors:"), dcc.Input(id='umap-neighbors-input', type='number', value=15, style={'width': '60px', 'marginLeft': '5px'})], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([html.Label("UMAP MinDist:"), dcc.Input(id='umap-mindist-input', type='number', value=0.1, step=0.05, style={'width': '60px', 'marginLeft': '5px'})], style={'display': 'inline-block', 'padding': '5px 10px'}),
            ], style={'textAlign': 'center'}),
            
            html.Div([
                dcc.Dropdown(id='color-selector', options=[
                    {'label': '채색: 자식 세력 점수', 'value': 'force_score'}, 
                    {'label': '채색: 부모 세력 점수', 'value': 'parent_force_score'}, # 부모 점수 채색 옵션 추가
                    {'label': '채색: 방향 (UP/DOWN)', 'value': 'direction'}
                ], value='force_score', clearable=False, style={'width': '250px', 'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '10px'}),
                dcc.RadioItems(id='scaling-method-selector', options=[{'label': '원시값', 'value': 'raw'}, {'label': '정규화', 'value': 'normalized'}, {'label': '로그 변환', 'value': 'log'}], value='raw', labelStyle={'display': 'inline-block', 'margin-right': '10px'}, style={'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '20px'}),
                html.Button('분석 및 시각화 실행', id='run-button', n_clicks=0, style={'padding': '8px 15px'}),
            ], style={'textAlign': 'center', 'marginTop': '10px'}),
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px'}),
    ], style={'textAlign': 'center', 'padding': '10px'}),

    dcc.Loading(id="loading-icon", children=[dcc.Graph(id='main-chart', style={'height': '60vh'})], type="default"),
    dcc.Loading(dcc.Graph(id='child-overview', style={'height': '45vh'})),
    html.Div(id='click-output', style={'textAlign': 'center', 'padding': '10px', 'fontSize': '16px', 'fontWeight': 'bold'}),
])

# ==============================================================================
# ## 섹션 3: Dash 콜백 (부모 정보 표시 로직 추가)
# ==============================================================================
@app.callback(
    Output('main-chart', 'figure'), Output('child-overview', 'figure'), Output('click-output', 'children'),
    Input('run-button', 'n_clicks'), Input('main-chart', 'clickData'),
    State('analysis-data-store', 'data'),
    State('ret-score-min', 'value'), State('ret-score-max', 'value'), State('pivot-min', 'value'), State('pivot-max', 'value'),
    State('slope-min', 'value'), State('slope-max', 'value'), State('direction-dropdown', 'value'),
    State('color-selector', 'value'), State('umap-neighbors-input', 'value'), State('umap-mindist-input', 'value'),
    State('scaling-method-selector', 'value'),
    prevent_initial_call=True
)
def universal_callback(run_clicks, clickData, analysis_data, rs_min, rs_max, p_min, p_max, s_min, s_max, direction,
                       color_mode, umap_neighbors, umap_min_dist, scaling_method):
    
    if not analysis_data:
        return go.Figure().update_layout(title_text='데이터 로드 실패', template='plotly_dark'), \
               go.Figure().update_layout(template='plotly_dark'), "데이터 로드 실패"
               
    df_parent_full = pd.DataFrame(analysis_data['parent_data'])
    df_child_full = pd.DataFrame(analysis_data['child_data_full'])

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'

    if triggered_id == 'main-chart' and clickData:
        clicked_child_id = clickData['points'][0]['customdata']
        clicked_child = df_child_full.loc[clicked_child_id]
        
        # 이제 자식 노드에서 바로 부모 ID를 알 수 있음
        parent_id = clicked_child['parent_id']
        if parent_id == -1:
            return dash.no_update, go.Figure().update_layout(title_text=f"오류: 자식 #{clicked_child_id}의 부모를 찾을 수 없음"), "부모 노드 검색 실패"
        
        P = df_parent_full.loc[parent_id]

        padding = (P['end_ts'] - P['start_ts']) * 1.5
        t0_ms, t1_ms = int(P['start_ts'] - padding), int(P['end_ts'] + padding)
        snap_hi = fetch_klines(SERVER_URL, SYMBOL, HI_TF, t0_ms, t1_ms)
        
        if snap_hi.empty:
            err_fig = go.Figure().update_layout(template='plotly_dark', title_text=f"오류: 서버에서 {HI_TF} 캔들 데이터를 받지 못했습니다.")
            return dash.no_update, err_fig, f"오류: {HI_TF} 캔들 조회 실패"

        fig = go.Figure(go.Candlestick(x=snap_hi.index, open=snap_hi['open'], high=snap_hi['high'], low=snap_hi['low'], close=snap_hi['close'], name=f'{HI_TF} ({SYMBOL})'))
        
        fig.add_vrect(x0=pd.to_datetime(P['start_ts'], unit='ms'), x1=pd.to_datetime(P['end_ts'], unit='ms'),
                      fillcolor='yellow', opacity=0.15, line_width=1, layer="below",
                      annotation_text=f"Parent (ID:{parent_id})", annotation_position="top left")

        sibling_kids = df_child_full[df_child_full['parent_id'] == parent_id]
        for _, k_row in sibling_kids.iterrows():
            color = 'rgba(0, 255, 0, 0.35)' if k_row['direction'] == 1.0 else 'rgba(255, 0, 0, 0.35)'
            fig.add_vrect(x0=pd.to_datetime(k_row['start_ts'], unit='ms'), x1=pd.to_datetime(k_row['end_ts'], unit='ms'),
                          fillcolor=color, line_width=0, layer='below')

        fig.add_vrect(x0=pd.to_datetime(clicked_child['start_ts'], unit='ms'), x1=pd.to_datetime(clicked_child['end_ts'], unit='ms'),
                      fillcolor="rgba(0,0,0,0)", line_width=2, line_color="cyan", layer='above',
                      annotation_text="Selected", annotation_font_color="cyan", annotation_position="bottom right")
        
        # [수정] 스냅샷 제목과 메시지에 부모 세력 점수 추가
        parent_force_score = P['force_score']
        msg = f"Parent #{parent_id} (Force: {parent_force_score:.2f}) | Clicked Child #{clicked_child_id} Highlighted"
        fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, title=f"Parent #{parent_id} Snapshot (Force: {parent_force_score:.2f}) | Child #{clicked_child_id} selected")
        return dash.no_update, fig, msg

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
        
        df_filtered = df_child_full.query(query_str) if query_parts else df_child_full
        
        if len(df_filtered) < 2:
            return go.Figure().update_layout(title_text='데이터가 너무 적습니다.', template='plotly_dark'), \
                   go.Figure().update_layout(template='plotly_dark'), "데이터 부족"

        features = df_filtered[['retracement_score', 'abs_angle_deg', 'pivot_count']].values
        reducer = umap.UMAP(n_neighbors=umap_neighbors, min_dist=umap_min_dist, random_state=42)
        embedding = reducer.fit_transform(StandardScaler().fit_transform(features))
        
        # [수정] hover text에 부모 세력 점수 추가
        hover_texts = [
            f"자식 ID: {idx}<br>"
            f"<b>자식 점수: {r['force_score']:.2f}</b><br>"
            f"부모 ID: {r['parent_id']}<br>"
            f"부모 점수: {r['parent_force_score']:.2f}<br>"
            f"방향: {'UP' if r['direction']==1.0 else 'DOWN'}" 
            for idx, r in df_filtered.iterrows()
        ]
        
        # 채색 로직
        cbar_title_base = '자식 세력 점수'
        scores_1d = df_filtered['force_score'].values
        colorscale = 'Viridis' # 기본값
        
        if color_mode == 'direction':
            marker_color, colorscale, cbar_title = df_filtered['direction'].map({1.0: 1, -1.0: 0}), [[0, 'lightcoral'], [1, 'lightgreen']], '방향'
        else: # 'force_score' 또는 'parent_force_score'
            if color_mode == 'parent_force_score':
                cbar_title_base = '부모 세력 점수'
                scores_1d = df_filtered['parent_force_score'].values
                colorscale = 'Plasma'
            else: # 'force_score'
                cbar_title_base = '자식 세력 점수'
                scores_1d = df_filtered['force_score'].values
                colorscale = 'Viridis'

            if scaling_method == 'normalized' and scores_1d.max() > scores_1d.min():
                marker_color, cbar_title = MinMaxScaler().fit_transform(scores_1d.reshape(-1, 1)).flatten(), f'{cbar_title_base} (정규화)'
            elif scaling_method == 'log':
                # 로그 변환은 0 또는 음수 값에 대해 이슈가 있을 수 있으므로 1p 사용
                marker_color, cbar_title = np.log1p(scores_1d - scores_1d.min()), f'{cbar_title_base} (로그 변환)'
            else: # 'raw'
                marker_color, cbar_title = scores_1d, cbar_title_base

        fig = go.Figure(data=[go.Scattergl(
            x=embedding[:, 0], y=embedding[:, 1], mode='markers', 
            customdata=df_filtered.index.to_list(),
            marker=dict(size=7, color=marker_color, colorscale=colorscale, showscale=True, opacity=0.8, colorbar={'title': cbar_title}),
            text=hover_texts, hoverinfo='text'
        )])
        fig.update_layout(template='plotly_dark', title_text=f'2D UMAP of Child Nodes (5m) | 데이터: {len(df_filtered)}개',
                          xaxis_title='UMAP 1', yaxis_title='UMAP 2', margin=dict(l=20, r=20, b=20, t=60))
        return fig, go.Figure().update_layout(template='plotly_dark', title="UMAP에서 자식 노드를 클릭하면 부모 컨텍스트 스냅샷이 표시됩니다."), f"필터링된 자식 노드: {len(df_filtered)}개"

    return dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    print("\n### UMAP 자식 노드(5m) 패턴 분석 플랫폼 (부모 정보 연계) ###")
    app.run(debug=True, port=8056)