# UMAP_Grandparent.py (Final - UMAP + Live K-means Clustering)

import dash
from dash import dcc, html
from dash import Input, Output, State, ctx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import traceback
import requests
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans # KMeans 라이브러리 추가
import umap

# ==============================================================================
# ## 섹션 0: 공통 설정
# ==============================================================================
SERVER_URL = "http://localhost:8202"
SYMBOL = "BTCUSDT"
TF_OPTIONS = ['5m', '15m', '1h', '4h', '1d']
NUM_CLUSTERS = 4 # ❗️ 클러스터 개수 설정 (여기서 변경 가능) ❗️

# ==============================================================================
# ## 섹션 1: 헬퍼 함수
# ==============================================================================

def get_parquet_path(timeframe):
    """타임프레임 문자열로 parquet 파일 경로를 생성"""
    if timeframe == '5m': return 'analysis_results_5years_robust.parquet'
    return f'{timeframe}_analysis_results_5years_robust.parquet'

def fetch_klines(server_url, symbol, timeframe, start_ms, end_ms):
    params = {"symbol": symbol, "timeframe": timeframe, "startTime": int(start_ms), "endTime": int(end_ms)}
    try:
        r = requests.get(f"{server_url}/api/klines", params=params, timeout=10)
        r.raise_for_status()
        js = r.json()
        if not js: return pd.DataFrame()
        df = pd.DataFrame(js, columns=['start_ts', 'open', 'high', 'low', 'close', 'volume'])
        df["ts"] = pd.to_datetime(df["start_ts"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
    except requests.exceptions.RequestException as e:
        print(f"서버로부터 캔들 데이터를 가져오는 데 실패했습니다: {e}")
        return pd.DataFrame()

def get_force_score(df):
    if 'direction' in df.columns:
        df['direction'] = df['direction'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
    return df['retracement_score'] * df['abs_angle_deg'].abs()

def load_and_enrich_data_with_grandparent(gp_path, p_path, c_path):
    """조부모-부모-자식 데이터 로드 및 2단계 보강 (방향 정보 추가)"""
    try:
        paths = {'gp': gp_path, 'p': p_path, 'c': c_path}
        for key, path in paths.items():
            if not os.path.exists(path):
                print(f"오류: {key} 분석 파일 '{path}'을 찾을 수 없습니다.")
                return None

        df_gp = pd.read_parquet(gp_path)
        df_p = pd.read_parquet(p_path)
        df_c = pd.read_parquet(c_path)

        df_gp['force_score'] = get_force_score(df_gp)
        df_p['force_score'] = get_force_score(df_p)
        df_c['force_score'] = get_force_score(df_c)
        
        # 1단계: 부모 -> 조부모 연결
        df_p['grandparent_id'] = -1
        for gp_id, gp_row in df_gp.iterrows():
            mask = (df_p['start_ts'] >= gp_row['start_ts']) & (df_p['end_ts'] <= gp_row['end_ts'])
            df_p.loc[mask, 'grandparent_id'] = gp_id

        # 2단계: 자식 -> 부모 -> 조부모 연결
        df_c['parent_id'] = -1
        df_c['parent_force_score'] = np.nan
        df_c['parent_direction'] = np.nan
        df_c['grandparent_id'] = -1
        df_c['grandparent_force_score'] = np.nan
        df_c['grandparent_direction'] = np.nan

        for p_id, p_row in df_p.iterrows():
            mask = (df_c['start_ts'] >= p_row['start_ts']) & (df_c['end_ts'] <= p_row['end_ts'])
            df_c.loc[mask, 'parent_id'] = p_id
            df_c.loc[mask, 'parent_force_score'] = p_row['force_score']
            df_c.loc[mask, 'parent_direction'] = p_row['direction']
            df_c.loc[mask, 'grandparent_id'] = p_row['grandparent_id'] 
        
        # 조부모 점수 및 방향 연결
        gp_scores = df_gp['force_score']
        gp_directions = df_gp['direction']
        df_c['grandparent_force_score'] = df_c['grandparent_id'].map(gp_scores)
        df_c['grandparent_direction'] = df_c['grandparent_id'].map(gp_directions)
        
        return {
            'grandparent_data': df_gp.to_dict('index'),
            'parent_data': df_p.to_dict('index'), 
            'child_data': df_c.to_dict('index')
        }
    except Exception as e:
        traceback.print_exc()
        return None

# ==============================================================================
# ## 섹션 2: Dash 앱 레이아웃
# ==============================================================================
app = dash.Dash(__name__)
app.title = "UMAP Analyzer (with Grandparent)"

app.layout = html.Div(style={'backgroundColor': '#1E1E1E', 'color': '#E0E0E0', 'fontFamily': 'sans-serif'}, children=[
    dcc.Store(id='analysis-data-store'),
    html.H1(children='UMAP 분석기 (조부모 계층 + 라이브 클러스터링)', style={'textAlign': 'center', 'padding': '15px'}),
    
    html.Div([
        html.Div([
            html.H4("0. 분석 대상 선택", style={'marginTop': '0'}),
            html.Div([
                html.Label("조부모 TF:", style={'marginRight': '5px'}),
                dcc.Dropdown(id='grandparent-tf-dropdown', options=TF_OPTIONS, value='4h', clearable=False, style={'width': '120px', 'color': '#1E1E1E'}),
            ], style={'display': 'inline-block', 'marginRight': '20px'}),
            html.Div([
                html.Label("부모 TF:", style={'marginRight': '5px'}),
                dcc.Dropdown(id='parent-tf-dropdown', options=TF_OPTIONS, value='1h', clearable=False, style={'width': '120px', 'color': '#1E1E1E'}),
            ], style={'display': 'inline-block', 'marginRight': '20px'}),
            html.Div([
                html.Label("자식 TF:", style={'marginRight': '5px'}),
                dcc.Dropdown(id='child-tf-dropdown', options=TF_OPTIONS, value='15m', clearable=False, style={'width': '120px', 'color': '#1E1E1E'}),
            ], style={'display': 'inline-block', 'marginRight': '30px'}),
            html.Button('데이터 로드', id='load-data-button', n_clicks=0, style={'padding': '8px 15px'}),
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px', 'marginBottom': '10px'}),
        
        html.Div([
            html.H4("1. 자식 노드 필터링", style={'marginTop': '0'}),
             html.Div([
                html.Div([html.Label("되돌림 점수:"), dcc.Input(id='ret-score-min', type='number'), dcc.Input(id='ret-score-max', type='number')], style={'display': 'inline-block', 'padding': '5px'}),
                html.Div([html.Label("피봇 개수:"), dcc.Input(id='pivot-min', type='number', value=4), dcc.Input(id='pivot-max', type='number', value=10)], style={'display': 'inline-block', 'padding': '5px'}),
                html.Div([html.Label("절대 각도:"), dcc.Input(id='slope-min', type='number'), dcc.Input(id='slope-max', type='number')], style={'display': 'inline-block', 'padding': '5px'}),
                html.Div([html.Label("방향:"), dcc.Dropdown(id='direction-dropdown', options=[{'label': '전체', 'value': 'all'}, {'label': 'UP', 'value': 'up'}, {'label': 'DOWN', 'value': 'down'}], value='all', clearable=False)], style={'display': 'inline-block', 'width': '150px', 'padding': '5px'}),
            ]),
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px', 'marginBottom': '10px'}),

        html.Div([
            html.H4("2. 분석 및 시각화 실행", style={'marginTop': '0'}),
            html.Div([
                html.Label("UMAP Neighbors:", style={'marginRight': '5px'}), dcc.Input(id='umap-neighbors-input', type='number', value=15, style={'width': '60px'}),
                html.Label("UMAP MinDist:", style={'marginLeft': '20px', 'marginRight': '5px'}), dcc.Input(id='umap-mindist-input', type='number', value=0.1, step=0.05, style={'width': '60px'}),
            ], style={'marginBottom': '10px'}),
            dcc.Dropdown(id='color-selector', options=[
                {'label': '채색: 자식 점수', 'value': 'force_score'}, 
                {'label': '채색: 부모 점수', 'value': 'parent_force_score'},
                {'label': '채색: 조부모 점수', 'value': 'grandparent_force_score'},
                {'label': '채색: 방향', 'value': 'direction'},
                {'label': '채색: 클러스터', 'value': 'cluster'}
            ], value='force_score', clearable=False, style={'width': '250px', 'marginRight': '10px'}),
            html.Button('분석 실행', id='run-button', n_clicks=0, style={'padding': '8px 15px'}),
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px'}),
    ], style={'textAlign': 'center', 'padding': '10px'}),

    dcc.Loading(dcc.Graph(id='main-chart', style={'height': '60vh'})),
    html.Div([
        dcc.Loading(dcc.Graph(id='parent-snapshot-chart', style={'height': '45vh', 'width': '50%', 'display': 'inline-block'})),
        dcc.Loading(dcc.Graph(id='grandparent-snapshot-chart', style={'height': '45vh', 'width': '50%', 'display': 'inline-block'})),
    ]),
    html.Div(id='click-output', style={'textAlign': 'center', 'padding': '10px'}),
])

# ==============================================================================
# ## 섹션 3: Dash 콜백
# ==============================================================================

@app.callback(
    Output('analysis-data-store', 'data'),
    Input('load-data-button', 'n_clicks'),
    [State('grandparent-tf-dropdown', 'value'), State('parent-tf-dropdown', 'value'), State('child-tf-dropdown', 'value')],
    prevent_initial_call=True
)
def load_data_callback(n_clicks, gp_tf, p_tf, c_tf):
    if not all([gp_tf, p_tf, c_tf]): return dash.no_update
    print(f"데이터 로드: 조부모={gp_tf}, 부모={p_tf}, 자식={c_tf}")
    gp_path, p_path, c_path = get_parquet_path(gp_tf), get_parquet_path(p_tf), get_parquet_path(c_tf)
    data = load_and_enrich_data_with_grandparent(gp_path, p_path, c_path)
    if data:
        data['grandparent_tf'] = gp_tf
        data['parent_tf'] = p_tf
        data['child_tf'] = c_tf
    return data

@app.callback(
    [Output('main-chart', 'figure'), Output('parent-snapshot-chart', 'figure'), Output('grandparent-snapshot-chart', 'figure'), Output('click-output', 'children')],
    [Input('run-button', 'n_clicks'), Input('main-chart', 'clickData')],
    [State('analysis-data-store', 'data'),
     State('ret-score-min', 'value'), State('ret-score-max', 'value'), State('pivot-min', 'value'), State('pivot-max', 'value'),
     State('slope-min', 'value'), State('slope-max', 'value'), State('direction-dropdown', 'value'),
     State('color-selector', 'value'), State('umap-neighbors-input', 'value'), State('umap-mindist-input', 'value')],
    prevent_initial_call=True
)
def universal_callback(run_clicks, clickData, analysis_data, rs_min, rs_max, p_min, p_max, s_min, s_max, direction,
                       color_mode, umap_neighbors, umap_min_dist):
    
    placeholder = go.Figure().update_layout(template='plotly_dark')
    if not analysis_data:
        placeholder.update_layout(title_text='분석할 타임프레임을 선택하고 [데이터 로드] 버튼을 누르세요.')
        return placeholder, placeholder, placeholder, "대기 중"

    try:
        gp_tf, p_tf, c_tf = analysis_data['grandparent_tf'], analysis_data['parent_tf'], analysis_data['child_tf']
        
        df_gp = pd.DataFrame.from_dict(analysis_data['grandparent_data'], orient='index')
        df_p = pd.DataFrame.from_dict(analysis_data['parent_data'], orient='index')
        df_c = pd.DataFrame.from_dict(analysis_data['child_data'], orient='index')

        for df in [df_gp, df_p, df_c]: df.index = pd.to_numeric(df.index)
        for col in ['parent_id', 'grandparent_id']:
            if col in df_c.columns: df_c[col] = pd.to_numeric(df_c[col], errors='coerce').fillna(-1).astype(int)
            if col in df_p.columns: df_p[col] = pd.to_numeric(df_p[col], errors='coerce').fillna(-1).astype(int)

    except (KeyError, TypeError) as e:
        placeholder.update_layout(title_text=f"데이터 구조 오류: {e}")
        return placeholder, placeholder, placeholder, "오류: 저장된 데이터가 올바르지 않습니다."

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'

    if triggered_id == 'main-chart' and clickData:
        try:
            clicked_child_id = int(clickData['points'][0]['customdata'])
            clicked_child = df_c.loc[clicked_child_id]
            parent_id = clicked_child['parent_id']
            grandparent_id = clicked_child['grandparent_id']
            
           # --- START: 클릭된 포인트 데이터 JSON으로 저장 ---
            save_message = ""
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                today_str = datetime.now().strftime('%Y-%m-%d')
                save_dir = os.path.join(script_dir, 'saved_data', today_str)
                
                os.makedirs(save_dir, exist_ok=True)
                file_name = f"clicked_point_{clicked_child_id}.json"
                file_path = os.path.join(save_dir, file_name)

                with open(file_path, 'w', encoding='utf-8') as f:
                    clicked_child.to_json(f, indent=4, force_ascii=False, default_handler=str)
                
                display_path = os.path.join('saved_data', today_str, file_name)
                save_message = f"Saved to {display_path}"
            except Exception as e:
                save_message = f"Save failed: {str(e)}"
                traceback.print_exc()
            # --- END: 파일 저장 로직 ---

            # 부모 스냅샷 생성
            if parent_id != -1:
                parent_row = df_p.loc[parent_id]
                padding = (parent_row['end_ts'] - parent_row['start_ts']) * 1.5
                df_candles_p = fetch_klines(SERVER_URL, SYMBOL, p_tf, parent_row['start_ts'] - padding, parent_row['end_ts'] + padding)
                
                fig_p = go.Figure(go.Candlestick(x=df_candles_p.index, open=df_candles_p.open, high=df_candles_p.high, low=df_candles_p.low, close=df_candles_p.close, name=p_tf))
                fig_p.add_vrect(x0=pd.to_datetime(parent_row.start_ts, unit='ms'), x1=pd.to_datetime(parent_row.end_ts, unit='ms'), fillcolor='yellow', opacity=0.1)
                fig_p.add_vrect(x0=pd.to_datetime(clicked_child.start_ts, unit='ms'), x1=pd.to_datetime(clicked_child.end_ts, unit='ms'), fillcolor='rgba(0,0,0,0)', line_color='cyan', line_width=2)
                fig_p.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, title=f"Parent ({p_tf}) #{parent_id}")
            else:
                fig_p = placeholder.update_layout(title_text="부모 없음")
            
            # 조부모 스냅샷 생성
            if grandparent_id != -1:
                gp_row = df_gp.loc[grandparent_id]
                padding = (gp_row['end_ts'] - gp_row['start_ts']) * 1.5
                df_candles_gp = fetch_klines(SERVER_URL, SYMBOL, gp_tf, gp_row['start_ts'] - padding, gp_row['end_ts'] + padding)

                fig_gp = go.Figure(go.Candlestick(x=df_candles_gp.index, open=df_candles_gp.open, high=df_candles_gp.high, low=df_candles_gp.low, close=df_candles_gp.close, name=gp_tf))
                fig_gp.add_vrect(x0=pd.to_datetime(gp_row.start_ts, unit='ms'), x1=pd.to_datetime(gp_row.end_ts, unit='ms'), fillcolor='yellow', opacity=0.1)
                fig_gp.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, title=f"Grandparent ({gp_tf}) #{grandparent_id}")
            else:
                fig_gp = placeholder.update_layout(title_text="조부모 없음")

            output_message = f"Child #{clicked_child_id} | {save_message}"
            return dash.no_update, fig_p, fig_gp, output_message

        except (KeyError, ValueError) as e:
            traceback.print_exc()
            return dash.no_update, placeholder.update_layout(title_text=f"스냅샷 오류: {e}"), placeholder, f"오류 발생: {e}"
            
    elif triggered_id == 'run-button':
        query_str = " and ".join([q for q in [
            f"retracement_score >= {rs_min}" if rs_min is not None else "", f"retracement_score <= {rs_max}" if rs_max is not None else "",
            f"pivot_count >= {p_min}" if p_min is not None else "", f"pivot_count <= {p_max}" if p_max is not None else "",
            f"abs_angle_deg >= {s_min}" if s_min is not None else "", f"abs_angle_deg <= {s_max}" if s_max is not None else "",
            f"direction == {1.0 if direction == 'up' else -1.0}" if direction != 'all' else ""
        ] if q])
        
        df_filtered = df_c.query(query_str) if query_str else df_c

        # ❗️ 라이브 클러스터링 실행 ❗️
        if len(df_filtered) >= NUM_CLUSTERS:
            features = df_filtered[['retracement_score', 'abs_angle_deg', 'pivot_count']].values
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init='auto')
            df_filtered.loc[:, 'cluster'] = kmeans.fit_predict(scaled_features)
        else:
            df_filtered.loc[:, 'cluster'] = np.nan
        
        if len(df_filtered) < 2:
            return placeholder.update_layout(title_text='데이터 부족'), placeholder, placeholder, "필터 조건에 맞는 데이터가 부족합니다."

        features = df_filtered[['retracement_score', 'abs_angle_deg', 'pivot_count']].values
        embedding = umap.UMAP(n_neighbors=umap_neighbors, min_dist=umap_min_dist, random_state=42).fit_transform(StandardScaler().fit_transform(features))
        
        _f = lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"
        _d = lambda d: 'UP' if d > 0 else ('DOWN' if d < 0 else 'N/A')
        
        hover_texts = [
            f"<b>ID: {idx} (D: {_d(r.direction)})</b><br>"
            f"Force: {_f(r.force_score)}<br>"
            f"Parent: {int(r.parent_id)} (F: {_f(r.parent_force_score)}, D: {_d(r.parent_direction)})<br>"
            f"G-Parent: {int(r.grandparent_id)} (F: {_f(r.grandparent_force_score)}, D: {_d(r.grandparent_direction)})"
            for idx, r in df_filtered.iterrows()
        ]

        if color_mode == 'direction':
            colors = df_filtered['direction'].map({1.0: 1, -1.0: 0})
            cbar_title, colorscale = '방향', [[0,'lightcoral'],[1,'lightgreen']]
        elif color_mode == 'cluster':
            colors = df_filtered['cluster'].astype(int)
            cbar_title, colorscale = '클러스터', 'Spectral'
            hover_texts = [f"{t}<br>Cluster: {int(c)}" for t, c in zip(hover_texts, colors)]
        else:
            colors = df_filtered[color_mode].fillna(0)
            cbar_title = {'parent_force_score': '부모 점수', 'grandparent_force_score': '조부모 점수'}.get(color_mode, '자식 점수')
            colorscale = 'Plasma' if 'parent' in color_mode else 'Viridis'

        umap_fig = go.Figure(go.Scattergl(
            x=embedding[:, 0], y=embedding[:, 1], mode='markers',
            customdata=df_filtered.index.astype(int).to_list(),
            marker=dict(color=colors, colorscale=colorscale, showscale=True, colorbar={'title': cbar_title}),
            text=hover_texts, hoverinfo='text'
        ))
        umap_fig.update_layout(template='plotly_dark', title=f'UMAP of {c_tf} patterns ({len(df_filtered)} items)')
        
        snap_placeholder = placeholder.update_layout(title_text="UMAP에서 점을 클릭하여 컨텍스트 확인")
        return umap_fig, snap_placeholder, snap_placeholder, f"분석 완료: {len(df_filtered)}개"

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    print("UMAP 분석기 (조부모 계층, 방향 정보, 데이터 저장 기능) 실행")
    app.run(debug=True, port=8057)