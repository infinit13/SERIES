# professional_analyzer_dashboard.py (Weighted UMAP + Snapshot Cluster - Complete Code)

import dash
from dash import dcc, html
from dash import Input, Output, State, ctx  # ctx가 callback_context 대체
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import mplfinance as mpf
import os
from datetime import datetime
import traceback

# --- 분석 라이브러리 임포트 ---
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import umap # umap-learn >= 0.5 필요

# ==============================================================================
# ## 섹션 1: 차트 스냅샷 생성 함수들 (전체 복원)
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

def visualize_single_series_and_save(df_kline, target_series, output_filename, title_prefix=""):
    """단일 시리즈에 대한 캔들 차트를 생성하고 파일로 저장합니다."""
    start_ms, end_ms = target_series['start_ts'], target_series['end_ts']
    title = f"{title_prefix} @ {pd.to_datetime(start_ms, unit='ms').strftime('%Y-%m-%d %H:%M')}"
    
    mc = mpf.make_marketcolors(up='#ff7f0e', down='#1f77b4', inherit=True) # 주황/파랑
    style = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='nightclouds', gridcolor='#363c4e', y_on_right=True)
    
    try:
        mpf.plot(df_kline, type='candle', style=style,
                 title=title,
                 savefig=dict(fname=output_filename, dpi=150, pad_inches=0.25))
        print(f"   > 성공: 차트 저장 완료 '{output_filename}'")
    except Exception as e:
        print(f"   > 오류: 차트 시각화 중 문제 발생: {e}")

# ==============================================================================
# ## 섹션 2: Dash 앱 레이아웃
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
                        {'label': '채색: 부모 세력 점수', 'value': 'force_score'},
                        {'label': '채색: 방향 (UP/DOWN)', 'value': 'direction'},
                    ],
                    value='child_strength', clearable=False,
                    style={'width': '300px', 'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '20px'}
                ),
                html.Button('분석 및 시각화 실행', id='run-button', n_clicks=0, style={'padding': '8px 15px'}),
            ], style={'textAlign': 'center', 'marginTop': '10px'}),
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px'}),
    ], style={'textAlign': 'center', 'padding': '10px'}),

    dcc.Loading(id="loading-icon", children=[dcc.Graph(id='main-chart', style={'height': '70vh'})], type="default"),
    html.Div(id='click-output', style={'textAlign': 'center', 'padding': '15px', 'fontSize': '16px', 'fontWeight': 'bold'}),
])

# ==============================================================================
# ## 섹션 3: Dash 콜백
# ==============================================================================

def get_force_score(df):
    """세력 점수를 계산하는 헬퍼 함수"""
    return df['retracement_score'] * df['abs_angle_deg']

@app.callback(
    Output('analysis-data-store', 'data'),
    Input('load-data-button', 'n_clicks'),
    State('parent-filepath-input', 'value'),
    State('child-filepath-input', 'value'),
    prevent_initial_call=True
)
def load_analysis_data(n_clicks, parent_path, child_path):
    """두 Parquet 파일을 로드하고 부모-자식 관계를 전처리하여 dcc.Store에 저장합니다."""
    print("\n\n--- '데이터 불러오기' 콜백 시작 ---")
    if not parent_path or not child_path:
        print(">>> 오류: 파일 경로가 하나 이상 비어있습니다.")
        return None
    try:
        print(f"1. 부모 파일 로딩 시도: '{parent_path}'")
        df_parent = pd.read_parquet(parent_path)
        print(f"--> 성공: 부모 파일 로딩 완료. (총 {len(df_parent)}개 행)")

        print(f"2. 자식 파일 로딩 시도: '{child_path}'")
        df_child = pd.read_parquet(child_path)
        print(f"--> 성공: 자식 파일 로딩 완료. (총 {len(df_child)}개 행)")

        print("3. 세력 점수 계산 중...")
        df_parent['force_score'] = get_force_score(df_parent)
        df_child['force_score'] = get_force_score(df_child)
        print("--> 성공: 세력 점수 계산 완료.")

        print("4. 부모-자식 매칭 및 가중치(y) 생성 시작...")
        child_max_force = []
        for parent_row in df_parent.itertuples(): # itertuples가 더 빠름
            children_in_range = df_child[
                (df_child['start_ts'] >= parent_row.start_ts) & 
                (df_child['end_ts'] <= parent_row.end_ts)
            ]
            child_max_force.append(children_in_range['force_score'].max() if not children_in_range.empty else 0)
        
        df_parent['child_max_force'] = child_max_force
        print("--> 성공: 모든 부모에 대한 '자식 최대 세력 점수' 계산 완료.")

        print("5. 자식 세력 등급(y) 생성 중...")
        strong_children_mask = df_parent['child_max_force'] > 0
        y = pd.Series(-1, index=df_parent.index, dtype=int)
        
        if strong_children_mask.any():
            print("   ... QuantileTransformer로 등급 분류 시도 ...")
            qt = QuantileTransformer(n_quantiles=5, output_distribution='uniform', random_state=42)
            scores_to_transform = df_parent.loc[strong_children_mask, ['child_max_force']]
            if scores_to_transform.nunique().iloc[0] < 5:
                 print("   > 경고: 고유한 자식 세력 점수 값이 5개 미만이라 등급을 줄입니다.")
                 qt.n_quantiles = scores_to_transform.nunique().iloc[0]

            transformed_scores = qt.fit_transform(scores_to_transform)
            y[strong_children_mask] = pd.cut(transformed_scores.flatten(), bins=qt.n_quantiles, labels=False, include_lowest=True)
            print("--> 성공: 등급 분류 완료.")
        else:
            print("--> 정보: 강력한 자식을 가진 부모가 없어 등급을 매기지 않음.")

        df_parent['y_child_strength_label'] = y
        print(f"6. 최종 데이터 생성 완료. 등급 분포:\n{y.value_counts().sort_index()}")
        
        return {
            'parent_data': df_parent.to_dict('records'),
            'child_data_full': df_child.to_dict('records')
        }
    except Exception as e:
        print("\n❗️❗️❗️ 데이터 로딩 및 전처리 중 치명적 오류 발생 ❗️❗️❗️")
        traceback.print_exc()
        return None

@app.callback(
    Output('main-chart', 'figure'),
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
    prevent_initial_call=True
)
def universal_callback(
    run_clicks, clickData, analysis_data,
    rs_min, rs_max, p_min, p_max, s_min, s_max, direction,
    color_mode, umap_neighbors, umap_min_dist, gamma
):
    # ctx = callback_context  <- 이 줄을 삭제!
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'

    if not analysis_data or 'parent_data' not in analysis_data:
    # ... 이하 코드는 동일 ...
        return go.Figure().update_layout(title_text="데이터를 먼저 불러오세요.", template='plotly_dark'), "대기 중..."

    df_parent_full = pd.DataFrame(analysis_data['parent_data'])
    df_parent_full.set_index(pd.Index(range(len(df_parent_full))), inplace=True) # 원본 인덱스 보존
    message = dash.no_update
    fig = dash.no_update

    # --- 스냅샷 클러스터 생성 로직 ---
    if triggered_id == 'main-chart' and clickData:
        try:
            original_index = clickData['points'][0]['customdata']
            clicked_parent_series = df_parent_full.loc[original_index]
            df_child_full = pd.DataFrame(analysis_data['child_data_full'])

            folder_name = f"snapshot_cluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}_parent_{original_index}"
            os.makedirs(folder_name, exist_ok=True)
            print(f"\n--- 클러스터 스냅샷 생성 시작: ./{folder_name} ---")

            parent_start_ts, parent_end_ts = clicked_parent_series['start_ts'], clicked_parent_series['end_ts']
            
            print(f"1. 부모 패턴 #{original_index} 스냅샷 생성 중...")
            parent_kline_padding = (parent_end_ts - parent_start_ts) * 0.5
            parent_kline = fetch_klines("BTCUSDT", "15m", int(parent_start_ts - parent_kline_padding), int(parent_end_ts + parent_kline_padding))
            if not parent_kline.empty:
                parent_output_path = os.path.join(folder_name, f"parent_{original_index}.png")
                visualize_single_series_and_save(parent_kline, clicked_parent_series, parent_output_path, "Parent")
            
            children_df = df_child_full[(df_child_full['start_ts'] >= parent_start_ts) & (df_child_full['end_ts'] <= parent_end_ts)]
            print(f"2. 총 {len(children_df)}개의 자식 패턴 스냅샷 생성 중...")
            for i, (child_index, child_series) in enumerate(children_df.iterrows()):
                child_start_ts, child_end_ts = child_series['start_ts'], child_series['end_ts']
                child_kline_padding = (child_end_ts - child_start_ts) * 0.5
                child_kline = fetch_klines("BTCUSDT", "5m", int(child_start_ts - child_kline_padding), int(child_end_ts + child_kline_padding))
                if not child_kline.empty:
                    child_output_path = os.path.join(folder_name, f"child_{i+1}_original_{child_index}.png")
                    visualize_single_series_and_save(child_kline, child_series, child_output_path, f"Child {i+1}")
            
            message = f"✅ 부모 1개와 자식 {len(children_df)}개의 스냅샷을 '{folder_name}' 폴더에 저장했습니다."
        except Exception as e:
            message = "스냅샷 생성 중 오류 발생!"
            print(f"클릭 이벤트 처리 중 오류: {e}")
            traceback.print_exc()

    # --- UMAP 분석 및 시각화 로직 ---
    elif triggered_id == 'run-button':
        query_parts = []
        if rs_min is not None: query_parts.append(f"retracement_score >= {rs_min}")
        if rs_max is not None: query_parts.append(f"retracement_score <= {rs_max}")
        if p_min is not None: query_parts.append(f"pivot_count >= {p_min}")
        if p_max is not None: query_parts.append(f"pivot_count <= {p_max}")
        if s_min is not None: query_parts.append(f"abs_angle_deg >= {s_min}")
        if s_max is not None: query_parts.append(f"abs_angle_deg <= {s_max}")
        if direction != 'all': query_parts.append(f"direction == {1.0 if direction == 'up' else -1.0}")
        
        df_filtered = df_parent_full.query(" and ".join(query_parts)) if query_parts else df_parent_full
        
        if df_filtered.empty:
            return go.Figure().update_layout(title_text='필터 조건에 맞는 데이터가 없습니다.', template='plotly_dark'), "조건에 맞는 데이터 없음"

        message = f"필터링된 데이터: {len(df_filtered)}개, Gamma: {gamma}"
        features = df_filtered[['retracement_score', 'abs_angle_deg', 'pivot_count']].values
        scaled_features = StandardScaler().fit_transform(features)
        y_supervised = df_filtered['y_child_strength_label'].values

        hover_texts = [f"인덱스: {idx}<br>부모 점수: {row['force_score']:.2f}<br>자식 최대점수: {row['child_max_force']:.2f}<br><b>자식 등급: {row['y_child_strength_label']}</b>" for idx, row in df_filtered.iterrows()]

        reducer = umap.UMAP(n_neighbors=umap_neighbors, min_dist=umap_min_dist, n_components=2, random_state=42, target_weight=gamma)
        print(f"Supervised UMAP을 실행합니다... (target_weight={gamma})")
        embedding = reducer.fit_transform(scaled_features, y=y_supervised)
        
        marker_color = df_filtered['force_score']
        colorscale = 'Viridis'
        if color_mode == 'child_strength':
            marker_color = df_filtered['y_child_strength_label']
            colorscale = 'Plasma'
        elif color_mode == 'direction':
            marker_color = ['lightgreen' if d == 1.0 else 'lightcoral' for d in df_filtered['direction']]
            colorscale = None
        
        fig = go.Figure(data=[go.Scattergl(
            x=embedding[:, 0], y=embedding[:, 1],
            mode='markers',
            customdata=df_filtered.index,
            marker=dict(size=7, color=marker_color, colorscale=colorscale, showscale=True if colorscale else False, opacity=0.8, colorbar={'title': color_mode}),
            text=hover_texts, hoverinfo='text'
        )])
        fig.update_layout(template='plotly_dark', margin=dict(l=20, r=20, b=20, t=60), title_text=f'2D UMAP 사영 | {message}', xaxis_title='UMAP 1', yaxis_title='UMAP 2')

    return fig, message

# ==============================================================================
# ## 섹션 4: 애플리케이션 실행
# ==============================================================================
if __name__ == '__main__':
    print("\n### 가중 UMAP 탐색적 패턴 분석 플랫폼 사용 안내 ###")
    print("1. 부모/자식 Parquet 파일이 스크립트와 동일한 디렉토리에 있는지 확인하세요.")
    print("2. 데이터 서버(Klines API)가 실행 중인지 확인하세요 (스냅샷 기능 사용 시).")
    print("3. 스크립트 실행 후 웹 브라우저에서 http://127.0.0.1:8052 주소로 접속하세요.")
    
    app.run(debug=True, port=8052)
