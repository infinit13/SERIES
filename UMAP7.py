# UMAP_Final_Integrated.py
# Gemini's Final Version with Lithography Scores, Hierarchies, and Interactive UI

import dash
from dash import dcc, html
from dash import Input, Output, State, ctx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import traceback
import requests
import os
from datetime import datetime

# --- 분석 라이브러리 임포트 ---
from sklearn.preprocessing import StandardScaler
import umap # umap-learn >= 0.5 필요

# ==============================================================================
# ## 섹션 0: 공통 설정
# ==============================================================================
SERVER_URL = "http://localhost:8202"
SYMBOL = "BTCUSDT"
TF_OPTIONS = ['5m', '15m', '1h', '4h', '1d']
LITHO_JSON_PATH = 'analysis_results.json' # 리소그라피 점수 파일 경로

# ==============================================================================
# ## 섹션 1: 헬퍼 함수
# ==============================================================================

def get_parquet_path(timeframe):
    """타임프레임 문자열로 parquet 파일 경로를 생성"""
    if timeframe == '5m': return 'analysis_results_5years_robust.parquet'
    return f'{timeframe}_analysis_results_5years_robust.parquet'

def fetch_klines(server_url, symbol, timeframe, start_ms, end_ms):
    """서버로부터 캔들 데이터 요청"""
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
    """Force Score 계산"""
    if 'direction' in df.columns:
        df['direction'] = df['direction'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
    return df['retracement_score'] * df['abs_angle_deg'].abs()

def add_litho_stats(df_series, df_litho):
    """시리즈 데이터프레임에 리소그라피 통계를 추가하는 함수"""
    def calculate_series_stats(series_row):
        mask = (df_litho['p1_ts_ms'] >= series_row['start_ts']) & (df_litho['p1_ts_ms'] < series_row['end_ts'])
        count = mask.sum()
        score_sum = df_litho.loc[mask, 'litho_score'].sum()
        return pd.Series([count, score_sum], index=['litho_seq_count', 'litho_series_score'])

    df_series[['litho_seq_count', 'litho_series_score']] = df_series.apply(calculate_series_stats, axis=1)
    df_series['litho_weighted_score'] = (df_series['litho_seq_count'] * df_series['litho_series_score']) + df_series['litho_seq_count']
    return df_series

def load_and_enrich_data_with_all_scores(gp_tf, p_tf, c_tf, litho_json_path):
    """모든 데이터를 로드하고, 리소그라피 점수 및 계층 구조를 포함하여 최종 보강"""
    try:
        # 1. 리소그라피 JSON 데이터 로드
        print(f"리소그라피 시퀀스 데이터 로드: {litho_json_path}")
        if not os.path.exists(litho_json_path):
            print(f"경고: '{litho_json_path}' 파일 없음.")
            df_litho = pd.DataFrame(columns=['p1_ts_ms', 'litho_score', 'status'])
        else:
            df_litho = pd.read_json(litho_json_path)
            df_litho['p1_ts_ms'] = pd.to_datetime(df_litho['p1_time_kst']).astype(np.int64) // 10**6
            df_litho = df_litho.rename(columns={'score': 'litho_score'})
        
        # 2. Parquet 시리즈 파일 로드
        paths = {'gp': get_parquet_path(gp_tf), 'p': get_parquet_path(p_tf), 'c': get_parquet_path(c_tf)}
        dataframes = {}
        for key, path in paths.items():
            if not os.path.exists(path): 
                print(f"오류: {key} 분석 파일 '{path}'을 찾을 수 없습니다.")
                return None
            df = pd.read_parquet(path)
            df.attrs['name'] = key
            dataframes[key] = df
        df_gp, df_p, df_c = dataframes['gp'], dataframes['p'], dataframes['c']

        # 3. 각 시리즈별 통계 계산
        for df in [df_gp, df_p, df_c]:
            add_litho_stats(df, df_litho)
            df['force_score'] = get_force_score(df)

        # 4. 계층 구조 연결
        df_p['grandparent_id'] = -1
        for gp_id, gp_row in df_gp.iterrows():
            mask = (df_p['start_ts'] >= gp_row['start_ts']) & (df_p['end_ts'] <= gp_row['end_ts'])
            df_p.loc[mask, 'grandparent_id'] = gp_id

        df_c['parent_id'] = -1
        df_c['parent_grandparent_id'] = -1
        
        parent_map_columns = ['force_score', 'direction', 'litho_series_score', 'litho_seq_count', 'litho_weighted_score', 'grandparent_id']
        for col in parent_map_columns:
            df_c[f'parent_{col}'] = np.nan

        for p_id, p_row in df_p.iterrows():
            mask = (df_c['start_ts'] >= p_row['start_ts']) & (df_c['end_ts'] <= p_row['end_ts'])
            df_c.loc[mask, 'parent_id'] = p_id
            for col in parent_map_columns:
                target_col = f'parent_{col}' if col != 'grandparent_id' else 'parent_grandparent_id'
                df_c.loc[mask, target_col] = p_row[col]

        gp_map_columns = ['force_score', 'direction', 'litho_series_score', 'litho_seq_count', 'litho_weighted_score']
        gp_data_map = df_gp.to_dict('index')
        for col in gp_map_columns:
            df_c[f'grandparent_{col}'] = df_c['parent_grandparent_id'].map(lambda x: gp_data_map.get(x, {}).get(col))

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
app.title = "UMAP Final Analyzer"

app.layout = html.Div(style={'backgroundColor': '#1E1E1E', 'color': '#E0E0E0', 'fontFamily': 'sans-serif'}, children=[
    dcc.Store(id='analysis-data-store'),
    html.H1(children='UMAP 최종 분석기 (리소그래피 점수 통합)', style={'textAlign': 'center', 'padding': '15px'}),
    
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
                html.Div([html.Label("피봇 개수:"), dcc.Input(id='pivot-min', type='number'), dcc.Input(id='pivot-max', type='number')], style={'display': 'inline-block', 'padding': '5px'}),
                html.Div([html.Label("절대 각도:"), dcc.Input(id='slope-min', type='number'), dcc.Input(id='slope-max', type='number')], style={'display': 'inline-block', 'padding': '5px'}),
                html.Div([html.Label("방향:"), dcc.Dropdown(id='direction-dropdown', options=[{'label': '전체', 'value': 'all'}, {'label': 'UP', 'value': 'up'}, {'label': 'DOWN', 'value': 'down'}], value='all', clearable=False, style={'display': 'inline-block', 'width': '150px', 'color': '#1E1E1E'})], style={'display': 'inline-block', 'padding': '5px'}),
            ]),
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px', 'marginBottom': '10px'}),

        html.Div([
            html.H4("2. 분석 및 시각화 실행", style={'marginTop': '0'}),
            html.Div([
                html.Label("UMAP Neighbors:", style={'marginRight': '5px'}), dcc.Input(id='umap-neighbors-input', type='number', value=15, style={'width': '60px'}),
                html.Label("UMAP MinDist:", style={'marginLeft': '20px', 'marginRight': '5px'}), dcc.Input(id='umap-mindist-input', type='number', value=0.1, step=0.05, style={'width': '60px'}),
            ], style={'marginBottom': '10px'}),
            dcc.Dropdown(id='color-selector', options=[
                {'label': '채색: 자식 리소 가중 점수', 'value': 'litho_weighted_score'},
                {'label': '채색: 부모 리소 가중 점수', 'value': 'parent_litho_weighted_score'},
                {'label': '채색: 조부모 리소 가중 점수', 'value': 'grandparent_litho_weighted_score'},
                {'label': '채색: 자식 Force 점수', 'value': 'force_score'}, 
                {'label': '채색: 부모 Force 점수', 'value': 'parent_force_score'},
                {'label': '채색: 조부모 Force 점수', 'value': 'grandparent_force_score'},
                {'label': '채색: 방향', 'value': 'direction'}
            ], value='litho_weighted_score', clearable=False, style={'width': '250px', 'marginRight': '10px', 'display': 'inline-block', 'color': '#1E1E1E'}),
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
    data = load_and_enrich_data_with_all_scores(gp_tf, p_tf, c_tf, LITHO_JSON_PATH)
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
    
    placeholder = go.Figure().update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    if not analysis_data:
        placeholder.update_layout(title_text='분석할 타임프레임을 선택하고 [데이터 로드] 버튼을 누르세요.')
        return placeholder, placeholder, placeholder, "대기 중"

    try:
        gp_tf, p_tf, c_tf = analysis_data['grandparent_tf'], analysis_data['parent_tf'], analysis_data['child_tf']
        df_gp = pd.DataFrame.from_dict(analysis_data['grandparent_data'], orient='index')
        df_p = pd.DataFrame.from_dict(analysis_data['parent_data'], orient='index')
        df_c = pd.DataFrame.from_dict(analysis_data['child_data'], orient='index')
        for df in [df_gp, df_p, df_c]: df.index = pd.to_numeric(df.index)
        
    except (KeyError, TypeError) as e:
        return placeholder.update_layout(title_text=f"데이터 구조 오류: {e}"), placeholder, placeholder, "오류"

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'

    if triggered_id == 'main-chart' and clickData:
        # ... (이 부분은 수정 없음) ...
        try:
            clicked_child_id = int(clickData['points'][0]['customdata'])
            clicked_child = df_c.loc[clicked_child_id]
            parent_id = int(clicked_child['parent_id'])
            grandparent_id = int(clicked_child['parent_grandparent_id'])
            
            fig_p = placeholder.update_layout(title_text="부모 없음")
            if parent_id != -1 and parent_id in df_p.index:
                parent_row = df_p.loc[parent_id]
                padding = (parent_row['end_ts'] - parent_row['start_ts']) * 1.5
                df_candles_p = fetch_klines(SERVER_URL, SYMBOL, p_tf, parent_row['start_ts'] - padding, parent_row['end_ts'] + padding)
                fig_p = go.Figure(go.Candlestick(x=df_candles_p.index, open=df_candles_p.open, high=df_candles_p.high, low=df_candles_p.low, close=df_candles_p.close, name=p_tf))
                fig_p.add_vrect(x0=pd.to_datetime(parent_row.start_ts, unit='ms'), x1=pd.to_datetime(parent_row.end_ts, unit='ms'), fillcolor='yellow', opacity=0.1)
                fig_p.add_vrect(x0=pd.to_datetime(clicked_child.start_ts, unit='ms'), x1=pd.to_datetime(clicked_child.end_ts, unit='ms'), fillcolor='rgba(0,0,0,0)', line_color='cyan', line_width=2)
                fig_p.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, title=f"Parent ({p_tf}) #{parent_id}", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            fig_gp = placeholder.update_layout(title_text="조부모 없음")
            if grandparent_id != -1 and grandparent_id in df_gp.index:
                gp_row = df_gp.loc[grandparent_id]
                padding = (gp_row['end_ts'] - gp_row['start_ts']) * 1.5
                df_candles_gp = fetch_klines(SERVER_URL, SYMBOL, gp_tf, gp_row['start_ts'] - padding, gp_row['end_ts'] + padding)
                fig_gp = go.Figure(go.Candlestick(x=df_candles_gp.index, open=df_candles_gp.open, high=df_candles_gp.high, low=df_candles_gp.low, close=df_candles_gp.close, name=gp_tf))
                fig_gp.add_vrect(x0=pd.to_datetime(gp_row.start_ts, unit='ms'), x1=pd.to_datetime(gp_row.end_ts, unit='ms'), fillcolor='yellow', opacity=0.1)
                fig_gp.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, title=f"Grandparent ({gp_tf}) #{grandparent_id}", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            output_message = f"Child #{clicked_child_id} | Parent #{parent_id} | G-Parent #{grandparent_id}"
            return dash.no_update, fig_p, fig_gp, output_message

        except (KeyError, ValueError) as e:
            traceback.print_exc()
            return dash.no_update, placeholder.update_layout(title_text=f"스냅샷 오류: {e}"), placeholder, f"오류 발생: {e}"
            
    elif triggered_id == 'run-button':
        query_parts = []
        if rs_min is not None: query_parts.append(f"retracement_score >= {rs_min}")
        if rs_max is not None: query_parts.append(f"retracement_score <= {rs_max}")
        if p_min is not None: query_parts.append(f"pivot_count >= {p_min}")
        if p_max is not None: query_parts.append(f"pivot_count <= {p_max}")
        if s_min is not None: query_parts.append(f"abs_angle_deg >= {s_min}")
        if s_max is not None: query_parts.append(f"abs_angle_deg <= {s_max}")
        if direction == 'up': query_parts.append("direction == 1.0")
        if direction == 'down': query_parts.append("direction == -1.0")
        query_str = " and ".join(query_parts)
        
        df_filtered = df_c.query(query_str) if query_str else df_c
        
        if len(df_filtered) < 2:
            return placeholder.update_layout(title_text='데이터 부족'), placeholder, placeholder, "필터 조건에 맞는 데이터가 부족합니다."

        features = df_filtered[['retracement_score', 'abs_angle_deg', 'pivot_count']].values
        embedding = umap.UMAP(n_neighbors=umap_neighbors, min_dist=umap_min_dist, random_state=42).fit_transform(StandardScaler().fit_transform(features))
        
        _f = lambda v: f"{v:.2f}" if pd.notna(v) and v is not None else "N/A"
        _c = lambda v: f"{int(v)}" if pd.notna(v) and v is not None else "N/A"
        _d = lambda d: 'UP' if d > 0 else ('DOWN' if d < 0 else 'N/A')
        
        hover_texts = [
            f"<b>ID: {idx} (D: {_d(r.direction)})</b><br>"
            f"--------------------<br>"
            f"<b>Child (Self):</b><br>"
            f"  Force: {_f(r.force_score)}<br>"
            f"  L-Weighted: {_f(r.litho_weighted_score)} (Sum: {_f(r.litho_series_score)}, Count: {_c(r.litho_seq_count)})<br>"
            f"<b>Parent #{_c(r.parent_id)}:</b><br>"
            f"  Force: {_f(r.parent_force_score)} (D: {_d(r.parent_direction)})<br>"
            f"  L-Weighted: {_f(r.parent_litho_weighted_score)} (Sum: {_f(r.parent_litho_series_score)}, Count: {_c(r.parent_litho_seq_count)})<br>"
            f"<b>G-Parent #{_c(r.parent_grandparent_id)}:</b><br>"
            f"  Force: {_f(r.grandparent_force_score)} (D: {_d(r.grandparent_direction)})<br>"
            f"  L-Weighted: {_f(r.grandparent_litho_weighted_score)} (Sum: {_f(r.grandparent_litho_series_score)}, Count: {_c(r.grandparent_litho_seq_count)})"
            for idx, r in df_filtered.iterrows()
        ]
        
        # ### 🚨 채색 로직 수정 ###
        colors = df_filtered[color_mode].fillna(0)
        cbar_title = ' '.join(color_mode.split('_')).title()
        
        marker_dict = {
            'color': colors,
            'showscale': True,
            'colorbar': {'title': cbar_title}
        }

        if color_mode == 'direction':
            marker_dict['colorscale'] = [[0,'red'],[1,'green']]
        elif 'score' in color_mode:
            marker_dict['colorscale'] = 'RdYlGn'
            
            # --- 스펙트럼 축소 로직 ---
            color_data = df_filtered[color_mode].dropna()
            if len(color_data) > 2:
                # 하위 10%, 상위 90% 지점을 색상 스펙트럼의 최소/최대로 설정
                vmin = color_data.quantile(0.10)
                vmax = color_data.quantile(0.90)
                # vmin과 vmax가 같으면 (대부분의 값이 같으면) 약간의 범위를 줌
                if vmin == vmax:
                    vmin -= 0.5
                    vmax += 0.5
                marker_dict['cmin'] = vmin
                marker_dict['cmax'] = vmax

        umap_fig = go.Figure(go.Scattergl(
            x=embedding[:, 0], y=embedding[:, 1], mode='markers',
            customdata=df_filtered.index.astype(int).to_list(),
            marker=marker_dict,
            text=hover_texts, hoverinfo='text'
        ))
        umap_fig.update_layout(template='plotly_dark', title=f'UMAP of {c_tf} patterns ({len(df_filtered)} items)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        
        snap_placeholder = placeholder.update_layout(title_text="UMAP에서 점을 클릭하여 컨텍스트 확인")
        return umap_fig, snap_placeholder, snap_placeholder, f"분석 완료: {len(df_filtered)}개"

    return placeholder, placeholder, placeholder, "대기 중"

if __name__ == '__main__':
    print("UMAP 최종 분석기 실행")
    app.run(debug=True, port=8059)