# professional_analyzer_dashboard.py (UMAP + DBSCAN Integrated)

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import callback_context
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import mplfinance as mpf
import os
from datetime import datetime

# ❗️분석 라이브러리 임포트
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import umap

# ==============================================================================
# ## 섹션 1: 분석/시각화 함수들 (기존과 동일)
# ==============================================================================
# 이 섹션의 함수들은 변경 없음 (fetch_klines, find_pivots_optimized 등)
def fetch_klines(symbol: str, timeframe: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    url = "http://localhost:8202/api/klines"
    params = {"symbol": symbol.upper(), "interval": timeframe, "startTime": start_ts, "endTime": end_ts}
    print(f"데이터 서버에서 스크린샷용 데이터를 요청합니다 ({start_ts} ~ {end_ts})...")
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        return df
    except requests.exceptions.RequestException as e:
        print(f"데이터 가져오기 오류: {e}")
        return pd.DataFrame()

def find_pivots_optimized(df: pd.DataFrame, lookaround: int):
    if df.empty or len(df) < (2 * lookaround + 1): return []
    highs, lows = df['high'].values, df['low'].values
    timestamps = df.index.astype(np.int64) // 10**6
    rolling_max = df['high'].rolling(window=2 * lookaround + 1, center=True).max().dropna().values
    rolling_min = df['low'].rolling(window=2 * lookaround + 1, center=True).min().dropna().values
    start_idx = lookaround
    raw_pivots = []
    peak_indices = np.where(highs[start_idx:len(df)-lookaround] >= rolling_max)[0] + start_idx
    trough_indices = np.where(lows[start_idx:len(df)-lookaround] <= rolling_min)[0] + start_idx
    for idx in peak_indices: raw_pivots.append({'time': timestamps[idx], 'price': highs[idx], 'type': 'P'})
    for idx in trough_indices: raw_pivots.append({'time': timestamps[idx], 'price': lows[idx], 'type': 'T'})
    raw_pivots.sort(key=lambda x: x['time'])
    if not raw_pivots: return []
    consolidated = []
    i = 0
    while i < len(raw_pivots):
        group = [raw_pivots[i]]
        j = i + 1
        while j < len(raw_pivots) and raw_pivots[j]['type'] == group[0]['type']: group.append(raw_pivots[j]); j += 1
        consolidated.append(max(group, key=lambda x: x['price']) if group[0]['type'] == 'P' else min(group, key=lambda x: x['price']))
        i = j
    if not consolidated: return []
    final_pivots = [consolidated[0]]
    for i in range(1, len(consolidated)):
        if consolidated[i]['type'] != final_pivots[-1]['type']: final_pivots.append(consolidated[i])
    return final_pivots

def analyze_channel(pivots, all_pivots, df, tolerance, is_upward):
    p_type, t_type = ('T', 'P') if is_upward else ('P', 'T')
    primary_pivots = sorted([p for p in pivots if p['type'] == p_type], key=lambda x: x['time'])
    if len(primary_pivots) < 2 or (is_upward and primary_pivots[1]['price'] < primary_pivots[0]['price']) or (not is_upward and primary_pivots[1]['price'] > primary_pivots[0]['price']): return None
    p1, p2 = primary_pivots[0], primary_pivots[1]
    if p1['time'] == p2['time']: return None
    slope = (p2['price'] - p1['price']) / (p2['time'] - p1['time'])
    first_secondary = next((p for p in sorted(pivots, key=lambda x: x['time']) if p['type'] == t_type and p['time'] > p1['time']), None)
    if not first_secondary: return None
    breakthrough_secondary = next((p for p in all_pivots if p['type'] == t_type and p['time'] > first_secondary['time'] and ((is_upward and p['price'] > first_secondary['price']) or (not is_upward and p['price'] < first_secondary['price']))), None)
    if not breakthrough_secondary: return None
    df_after_p2 = df[df.index > pd.to_datetime(p2['time'], unit='ms')]
    if df_after_p2.empty: return None
    candle_times, lows, highs = df_after_p2.index.astype(np.int64) // 10**6, df_after_p2['low'].values, df_after_p2['high'].values
    main_boundaries = slope * (candle_times - p1['time']) + p1['price']
    parallel_boundaries = slope * (candle_times - breakthrough_secondary['time']) + breakthrough_secondary['price']
    lower_break = np.where(lows < (main_boundaries * (1 - tolerance)))[0] if is_upward else np.where(lows < (parallel_boundaries * (1 - tolerance)))[0]
    upper_break = np.where(highs > (parallel_boundaries * (1 + tolerance)))[0] if is_upward else np.where(highs > (main_boundaries * (1 + tolerance)))[0]
    break_idx = min(lower_break[0] if lower_break.size > 0 else float('inf'), upper_break[0] if upper_break.size > 0 else float('inf'))
    channel_end_time = candle_times[break_idx] if break_idx != float('inf') else float('inf')
    pivots_in_channel = [p for p in all_pivots if p['type'] == t_type and p1['time'] <= p['time'] < channel_end_time]
    if not pivots_in_channel: return None
    extreme_pivot = max(pivots_in_channel, key=lambda p: p['price']) if is_upward else min(pivots_in_channel, key=lambda p: p['price'])
    return {'x0': p1['time'], 'y0': p1['price'], 'x1': extreme_pivot['time'], 'y1': extreme_pivot['price']}

def find_main_series_optimized(all_pivots, df, tolerance):
    main_series_shapes, pivot_index = [], 0
    while pivot_index < len(all_pivots) - 2:
        current_pivot, next_pivot = all_pivots[pivot_index], all_pivots[pivot_index + 1]
        connecting_line = None
        pivot_context = all_pivots[pivot_index : pivot_index + 20]
        if current_pivot['type'] == 'T' and next_pivot['type'] == 'P':
            connecting_line = analyze_channel(pivot_context, all_pivots, df, tolerance, is_upward=True)
            if connecting_line: main_series_shapes.append({"type": "MAIN_UP", "shape": connecting_line})
        elif current_pivot['type'] == 'P' and next_pivot['type'] == 'T':
            connecting_line = analyze_channel(pivot_context, all_pivots, df, tolerance, is_upward=False)
            if connecting_line: main_series_shapes.append({"type": "MAIN_DOWN", "shape": connecting_line})
        if connecting_line:
            next_start_pivot_idx = next((i for i, p in enumerate(all_pivots) if p['time'] >= connecting_line['x1']), None)
            pivot_index = next_start_pivot_idx if next_start_pivot_idx is not None else len(all_pivots)
        else: pivot_index += 1
    return sorted(main_series_shapes, key=lambda s: s['shape']['x0'])

def build_hybrid_series_sequence(df, all_pivots, tolerance):
    main_series = find_main_series_optimized(all_pivots, df, tolerance)
    if not main_series:
        return [{"type": f"SUB_{'UP' if p2['price'] > p1['price'] else 'DOWN'}", "shape": {"x0": p1['time'], "y0": p1['price'], "x1": p2['time'], "y1": p2['price']}} for p1, p2 in zip(all_pivots, all_pivots[1:])]
    consolidated_series, last_time, pivot_map = [], 0, {p['time']: p for p in all_pivots}
    for s_obj in main_series:
        gap_pivots = [p for p in all_pivots if last_time <= p['time'] < s_obj['shape']['x0']]
        if last_time > 0 and pivot_map.get(last_time) and (not gap_pivots or gap_pivots[0]['time'] != last_time): gap_pivots.insert(0, pivot_map.get(last_time))
        for p1, p2 in zip(gap_pivots, gap_pivots[1:]): consolidated_series.append({"type": f"SUB_{'UP' if p2['price'] > p1['price'] else 'DOWN'}", "shape": {"x0": p1['time'], "y0": p1['price'], "x1": p2['time'], "y1": p2['price']}})
        consolidated_series.append(s_obj)
        last_time = s_obj['shape']['x1']
    remaining_pivots = [p for p in all_pivots if p['time'] >= last_time]
    for p1, p2 in zip(remaining_pivots, remaining_pivots[1:]): consolidated_series.append({"type": f"SUB_{'UP' if p2['price'] > p1['price'] else 'DOWN'}", "shape": {"x0": p1['time'], "y0": p1['price'], "x1": p2['time'], "y1": p2['price']}})
    main_coords = {(s['shape']['x0'], s['shape']['x1']) for s in main_series}
    return [s for s in consolidated_series if not (s['type'].startswith('SUB') and (s['shape']['x0'], s['shape']['x1']) in main_coords)]

def visualize_single_series_and_save(df, all_series, target_series, output_filename, all_pivots):
    start_ms, end_ms = target_series['shape']['x0'], target_series['shape']['x1']
    padding = (end_ms - start_ms) * 0.5
    plot_df = df[(df.index.astype(np.int64)//10**6 >= start_ms - padding) & (df.index.astype(np.int64)//10**6 <= end_ms + padding)]
    if plot_df.empty:
        print(f"경고: 해당 시리즈 기간에 대한 데이터가 없어 스크린샷을 건너뜁니다: {output_filename}")
        return

    series_to_plot = [s for s in all_series if s['shape']['x0'] >= plot_df.index.min().value//10**6 and s['shape']['x1'] <= plot_df.index.max().value//10**6]
    lines, colors, styles, widths = [], [], [], []
    for s in series_to_plot:
        lines.append([(pd.to_datetime(s['shape']['x0'], unit='ms'), s['shape']['y0']), (pd.to_datetime(s['shape']['x1'], unit='ms'), s['shape']['y1'])])
        is_main, is_target = 'MAIN' in s['type'], (s['shape']['x0'] == start_ms and s['shape']['x1'] == end_ms)
        base_color = 'gold' if is_target else ('yellow' if is_main else ('deepskyblue' if 'UP' in s['type'] else 'orangered'))
        color_map = {'gold': (1.0, 0.84, 0, 0.7), 'yellow': (1.0, 1.0, 0, 0.7), 'deepskyblue': (0, 0.75, 1.0, 0.6), 'orangered': (1.0, 0.27, 0, 0.6)}
        colors.append(color_map[base_color])
        styles.append('-' if (is_main or is_target) else '--')
        widths.append(2.5 if is_target else (2.0 if is_main else 1.5))

    pivots_in_range = [p for p in all_pivots if plot_df.index.min().value//10**6 <= p['time'] <= plot_df.index.max().value//10**6]
    high_pivots = [p for p in pivots_in_range if p['type'] == 'P']
    low_pivots = [p for p in pivots_in_range if p['type'] == 'T']
    high_pivot_markers = pd.Series(np.nan, index=plot_df.index)
    if high_pivots:
        high_pivot_times = [pd.to_datetime(p['time'], unit='ms') for p in high_pivots]
        high_pivot_prices = [p['price'] for p in high_pivots]
        high_pivot_markers.loc[high_pivot_times] = high_pivot_prices

    low_pivot_markers = pd.Series(np.nan, index=plot_df.index)
    if low_pivots:
        low_pivot_times = [pd.to_datetime(p['time'], unit='ms') for p in low_pivots]
        low_pivot_prices = [p['price'] for p in low_pivots]
        low_pivot_markers.loc[low_pivot_times] = low_pivot_prices
    addplots = []
    if not high_pivot_markers.dropna().empty:
        ap_high = mpf.make_addplot(high_pivot_markers, type='scatter', marker='v', color=(1.0, 0.2, 0.2, 0.6), markersize=60)
        addplots.append(ap_high)
    if not low_pivot_markers.dropna().empty:
        ap_low = mpf.make_addplot(low_pivot_markers, type='scatter', marker='^', color=(0.2, 0.8, 1.0, 0.6), markersize=60)
        addplots.append(ap_low)

    mc = mpf.make_marketcolors(up='darkorange', down='royalblue', inherit=True)
    style = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='nightclouds', gridcolor='#363c4e', y_on_right=True)
    
    try:
        mpf.plot(plot_df, type='candle', style=style,
                 title=f"Series Screenshot: {pd.to_datetime(start_ms, unit='ms').strftime('%Y-%m-%d %H:%M')}",
                 alines=dict(alines=lines, colors=colors, linestyle=styles, linewidths=widths),
                 addplot=addplots if addplots else None,
                 savefig=dict(fname=output_filename, dpi=150))
        print(f"성공! 단일 시리즈 차트가 '{output_filename}'에 저장되었습니다.")
    except Exception as e:
        print(f"단일 시리즈 차트 시각화 중 오류 발생: {e}")

# ==============================================================================
# ## 섹션 2: Dash 앱 레이아웃 및 컴포넌트 (전면 개편)
# ==============================================================================
app = dash.Dash(__name__)
app.title = "Exploratory Pattern Analysis Platform"

app.layout = html.Div(style={'backgroundColor': '#1E1E1E', 'color': '#E0E0E0', 'fontFamily': 'sans-serif'}, children=[
    dcc.Store(id='analysis-data-store'),
    html.H1(children='탐색적 패턴 분석 플랫폼 (UMAP + DBSCAN)', style={'textAlign': 'center', 'padding': '15px'}),
    
    # --- 컨트롤 패널 ---
    html.Div([
        # 데이터 로드
        html.Div([
            html.H4("1. 데이터 로드", style={'marginTop': '0', 'marginBottom': '10px'}),
            dcc.Input(id='filepath-input', type='text', placeholder='분석 결과 파일 경로', value='analysis_results_5years_robust.parquet', style={'width': '300px'}),
            html.Button('데이터 불러오기', id='load-data-button', n_clicks=0, style={'padding': '8px 15px', 'marginLeft': '10px'}),
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px', 'marginBottom': '10px'}),

        # 필터링
        html.Div([
            html.H4("2. 데이터 필터링", style={'marginTop': '0', 'marginBottom': '10px'}),
            html.Div([
                html.Div([html.Label("되돌림 점수:"), dcc.Input(id='ret-score-min', type='number', placeholder='최소'), dcc.Input(id='ret-score-max', type='number', placeholder='최대')], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([html.Label("피봇 개수:"), dcc.Input(id='pivot-min', type='number', placeholder='최소'), dcc.Input(id='pivot-max', type='number', placeholder='최대')], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([html.Label("절대 각도:"), dcc.Input(id='slope-min', type='number', placeholder='최소'), dcc.Input(id='slope-max', type='number', placeholder='최대')], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([html.Label("방향:"), dcc.Dropdown(id='direction-dropdown', options=[{'label': '전체', 'value': 'all'}, {'label': 'UP', 'value': 'up'}, {'label': 'DOWN', 'value': 'down'}], value='all', clearable=False)], style={'display': 'inline-block', 'width': '150px', 'padding': '5px 10px', 'verticalAlign': 'middle'}),
            ], style={'textAlign': 'center'}),
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px', 'marginBottom': '10px'}),

        # 분석 및 시각화 설정
        html.Div([
            html.H4("3. 분석 및 시각화 설정", style={'marginTop': '0', 'marginBottom': '10px'}),
            html.Div([
                # UMAP 파라미터
                html.Div([
                    html.Label("UMAP Neighbors:", title="이웃 수. 낮으면 국소 구조, 높으면 전역 구조 강조."),
                    dcc.Input(id='umap-neighbors-input', type='number', value=15, style={'width': '60px', 'marginLeft': '5px'})
                ], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([
                    html.Label("UMAP MinDist:", title="최소 거리. 낮으면 군집이 빽빽해지고, 높으면 느슨해짐."),
                    dcc.Input(id='umap-mindist-input', type='number', value=0.1, step=0.05, style={'width': '60px', 'marginLeft': '5px'})
                ], style={'display': 'inline-block', 'padding': '5px 10px'}),
                # DBSCAN 파라미터
                html.Div([
                    html.Label("DBSCAN Eps:", title="탐색 반경. UMAP/3D 스케일에 따라 조절 필요."),
                    dcc.Input(id='dbscan-eps-input', type='number', value=0.5, step=0.05, style={'width': '60px', 'marginLeft': '5px'})
                ], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([
                    html.Label("DBSCAN MinSamples:", title="군집 인정을 위한 최소 포인트 수."),
                    dcc.Input(id='dbscan-minsamples-input', type='number', value=5, style={'width': '60px', 'marginLeft': '5px'})
                ], style={'display': 'inline-block', 'padding': '5px 10px'}),
            ], style={'textAlign': 'center'}),
            
            html.Div([
                dcc.RadioItems(
                    id='view-selector',
                    options=[{'label': '3D 원본 공간', 'value': '3d'}, {'label': '2D UMAP 사영', 'value': 'umap'}],
                    value='3d',
                    labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                    style={'padding': '10px 0'}
                ),
                dcc.Dropdown(
                    id='color-selector',
                    options=[
                        {'label': '채색: 방향 (UP/DOWN)', 'value': 'direction'},
                        {'label': '채색: DBSCAN 클러스터', 'value': 'dbscan'}
                    ],
                    value='direction',
                    clearable=False,
                    style={'width': '250px', 'display': 'inline-block', 'verticalAlign': 'middle'}
                ),
                html.Button('분석 및 시각화 실행', id='run-button', n_clicks=0, style={'padding': '8px 15px', 'marginLeft': '20px'}),
            ], style={'textAlign': 'center', 'marginTop': '10px'}),
            
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px'}),
        
    ], style={'textAlign': 'center', 'padding': '10px'}),
    
    # --- 차트 및 결과 ---
    dcc.Loading(id="loading-icon", children=[dcc.Graph(id='main-chart', style={'height': '70vh'})], type="default"),
    html.Div(id='click-output', style={'textAlign': 'center', 'padding': '15px', 'fontSize': '16px', 'fontWeight': 'bold'}),

    # 스크린샷용 숨겨진 파라미터
    dcc.Input(id='lookaround-input', type='number', value=5, style={'display': 'none'}),
    dcc.Input(id='tolerance-input', type='number', value=0.001, style={'display': 'none'}),
])


# ==============================================================================
# ## 섹션 3: Dash 콜백 (전면 개편)
# ==============================================================================

@app.callback(
    Output('analysis-data-store', 'data'),
    Input('load-data-button', 'n_clicks'),
    State('filepath-input', 'value'),
    prevent_initial_call=True
)
def load_analysis_data(n_clicks, filepath):
    if not filepath: return None
    try:
        # 올바른 컬럼 이름으로 데이터프레임 로드
        df = pd.read_parquet(filepath)
        # 필요한 컬럼만 선택하고 순서 고정
        columns = ['start_ts', 'end_ts', 'retracement_score', 'pivot_count', 'abs_angle_deg', 'direction']
        df = df[columns]
        vectors_with_indices = list(enumerate(df.values.tolist()))
        print(f"'{filepath}'에서 {len(vectors_with_indices)}개의 데이터를 성공적으로 불러왔습니다.")
        return {'vectors': vectors_with_indices}
    except Exception as e:
        print(f"파일 로드 오류: {e}")
        return None


@app.callback(
    Output('main-chart', 'figure'),
    Output('click-output', 'children'),
    Input('run-button', 'n_clicks'),
    Input('main-chart', 'clickData'),
    State('analysis-data-store', 'data'),
    # 필터
    State('ret-score-min', 'value'), State('ret-score-max', 'value'),
    State('pivot-min', 'value'), State('pivot-max', 'value'),
    State('slope-min', 'value'), State('slope-max', 'value'),
    State('direction-dropdown', 'value'),
    # 분석/시각화 설정
    State('view-selector', 'value'),
    State('color-selector', 'value'),
    State('umap-neighbors-input', 'value'),
    State('umap-mindist-input', 'value'),
    State('dbscan-eps-input', 'value'),
    State('dbscan-minsamples-input', 'value'),
    # 스크린샷용
    State('lookaround-input', 'value'),
    State('tolerance-input', 'value'),
    prevent_initial_call=True
)
def universal_callback(
    run_clicks, clickData, analysis_data,
    rs_min, rs_max, p_min, p_max, s_min, s_max, direction,
    view_mode, color_mode,
    umap_neighbors, umap_min_dist, dbscan_eps, dbscan_min_samples,
    lookaround, tolerance
):
    if not analysis_data or 'vectors' not in analysis_data:
        return go.Figure().update_layout(title_text="먼저 '데이터 불러오기'를 실행하세요.", template='plotly_dark'), "대기 중..."

    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'

    # 1. 데이터 필터링
    vectors_with_indices = analysis_data['vectors']
    filtered_vectors_with_indices = [
        item for item in vectors_with_indices
        if (rs_min is None or item[1][2] >= rs_min) and (rs_max is None or item[1][2] <= rs_max) and
           (p_min is None or item[1][3] >= p_min) and (p_max is None or item[1][3] <= p_max) and
           (s_min is None or item[1][4] >= s_min) and (s_max is None or item[1][4] <= s_max) and
           (direction == 'all' or (direction == 'up' and item[1][5] == 1.0) or (direction == 'down' and item[1][5] == -1.0))
    ]
    
    message = f"필터링된 데이터: {len(filtered_vectors_with_indices)}개"

    # 2. 클릭 이벤트 처리 (스크린샷)
    if triggered_id == 'main-chart' and clickData:
        point = clickData['points'][0]
        original_index = point['customdata']
        original_vector = next((item[1] for item in vectors_with_indices if item[0] == original_index), None)

        if original_vector:
            start_ts, end_ts = int(original_vector[0]), int(original_vector[1])
            padding = (end_ts - start_ts) * 1.0
            print(f"클릭된 시리즈({original_index})의 상세 분석을 시작합니다...")
            df_local = fetch_klines("BTCUSDT", "5m", int(start_ts - padding), int(end_ts + padding))
            if not df_local.empty:
                local_pivots = find_pivots_optimized(df_local, lookaround)
                local_series_sequence = build_hybrid_series_sequence(df_local, local_pivots, tolerance)
                target_series = min(local_series_sequence, key=lambda s: abs(s['shape']['x0'] - start_ts) + abs(s['shape']['x1'] - end_ts))
                output_filename = f"screenshot_{datetime.fromtimestamp(start_ts/1000).strftime('%Y%m%d_%H%M%S')}_series_{original_index}.png"
                visualize_single_series_and_save(df_local, local_series_sequence, target_series, output_filename, local_pivots)
                message += f" | ✅ 스크린샷 저장 완료: {output_filename}"
            else: message += " | ❌ 스크린샷 생성 실패: 데이터 로드 오류."
        else: message += " | ❌ 스크린샷 생성 실패: 원본 벡터 찾기 오류."
    
    # 3. 시각화 로직
    if not filtered_vectors_with_indices:
        return go.Figure().update_layout(title_text='필터 조건에 맞는 데이터가 없습니다.', template='plotly_dark'), message

    original_indices, filtered_vectors = zip(*filtered_vectors_with_indices)
    df_filtered = pd.DataFrame([list(v) for v in filtered_vectors], columns=['start_ts', 'end_ts', 'retracement_score', 'pivot_count', 'abs_angle_deg', 'direction'])
    
    # 분석에 사용할 특징 선택
    features = df_filtered[['retracement_score', 'abs_angle_deg', 'pivot_count']].values
    
    # 데이터 스케일링 (필수!)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 기본 색상 및 호버 텍스트 설정
    colors = ['lightgreen' if v[5] == 1.0 else 'lightcoral' for v in filtered_vectors]
    hover_texts = [f"인덱스: {idx}<br>점수: {v[2]:.2f}<br>각도: {v[4]:.2f}°<br>피봇: {v[3]}<br>방향: {'UP' if v[5] == 1.0 else 'DOWN'}" for idx, v in zip(original_indices, filtered_vectors)]
    
    fig = go.Figure()
    fig.update_layout(template='plotly_dark', margin=dict(l=10, r=20, b=10, t=60))
    
    # --- 채색 기준 분기 ---
    if color_mode == 'dbscan':
        # DBSCAN을 위한 데이터 준비 (뷰 모드에 따라 다름)
        data_for_dbscan = scaled_features if view_mode == '3d' else umap.UMAP(n_neighbors=umap_neighbors, min_dist=umap_min_dist, random_state=42).fit_transform(scaled_features)
        
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = dbscan.fit_predict(data_for_dbscan)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        message += f" | DBSCAN 결과: {n_clusters}개 클러스터, {n_noise}개 노이즈"
        
        # 클러스터별 색상 지정
        unique_labels = sorted(list(set(labels)))
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        label_colors = {label: color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels) if label != -1}
        label_colors[-1] = '#444444' # 노이즈 색상
        colors = [label_colors[l] for l in labels]
        
        # 호버 텍스트에 클러스터 정보 추가
        hover_texts = [f"{ht}<br>클러스터: {l}" for ht, l in zip(hover_texts, labels)]

    # --- 뷰 모드 분기 ---
    if view_mode == '3d':
        fig.add_trace(go.Scatter3d(
            x=df_filtered['retracement_score'], y=df_filtered['abs_angle_deg'], z=df_filtered['pivot_count'],
            mode='markers', customdata=original_indices,
            marker=dict(size=5, symbol='square', color=colors, opacity=0.8),
            text=hover_texts, hoverinfo='text'
        ))
        fig.update_layout(
            title_text=f'3D 벡터 공간 | {message}',
            scene=dict(xaxis_title='되돌림 점수', yaxis_title='절대 각도 (도)', zaxis_title='피봇 개수')
        )
    
    elif view_mode == 'umap':
        reducer = umap.UMAP(n_neighbors=umap_neighbors, min_dist=umap_min_dist, n_components=2, random_state=42)
        embedding = reducer.fit_transform(scaled_features)
        
        fig.add_trace(go.Scatter(
            x=embedding[:, 0], y=embedding[:, 1],
            mode='markers', customdata=original_indices,
            marker=dict(size=7, color=colors, opacity=0.8),
            text=hover_texts, hoverinfo='text'
        ))
        fig.update_layout(
            title_text=f'2D UMAP 사영 | {message}',
            xaxis_title='UMAP 1', yaxis_title='UMAP 2'
        )

    return fig, message


# ==============================================================================
# ## 섹션 4: 애플리케이션 실행
# ==============================================================================
if __name__ == '__main__':
    print("\n### 탐색적 패턴 분석 플랫폼 사용 안내 ###")
    print("1. 데이터 서버(server_5min.cjs)가 실행 중인지 확인하세요.")
    print("2. 분석 결과 파일(.parquet)이 현재 디렉토리에 있는지 확인하세요.")
    print("3. 이 스크립트를 실행한 뒤 웹 브라우저에서 http://127.0.0.1:8081 주소로 접속하세요.")
    print("4. '데이터 불러오기' -> 원하는 필터/파라미터 설정 -> '분석 및 시각화 실행' 버튼을 누르세요.")
   
    app.run(debug=True, port=8081)