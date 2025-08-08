# professional_analyzer_dashboard.py (Timeframe-aware Screenshot Frame Added)

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

# ==============================================================================
# ## 섹션 1: 분석/시각화 함수들 (수정됨)
# ==============================================================================
def fetch_klines(symbol: str, timeframe: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """지정한 기간의 캔들 데이터를 서버에서 가져옵니다."""
    url = "http://localhost:8202/api/klines"
    params = {"symbol": symbol.upper(), "interval": timeframe, "startTime": start_ts, "endTime": end_ts}
    print(f"데이터 서버에서 스크린샷용 데이터를 요청합니다 (Timeframe: {timeframe}, Range: {start_ts} ~ {end_ts})...")
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
    """최적화된 피봇 찾기 함수 (기존과 동일)"""
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
    """채널 분석 함수 (기존과 동일)"""
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
    """메인 시리즈 찾기 함수 (기존과 동일)"""
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
    """하이브리드 시퀀스 생성 함수 (기존과 동일)"""
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

def visualize_single_series_and_save(df, all_series, target_series, output_filename, all_pivots, timeframe="5m"):
    """❗️❗️❗️ 차트 시각화 및 저장 (타임프레임에 따른 프레임 조절 기능 추가) ❗️❗️❗️"""
    start_ms, end_ms = target_series['shape']['x0'], target_series['shape']['x1']

    # 타임프레임별로 차트의 좌우 여백(Padding)을 다르게 설정
    padding_map = {'1m': 1.0, '5m': 0.5, '15m': 0.3}
    padding_multiplier = padding_map.get(timeframe, 0.5) # 기본값 0.5
    padding = (end_ms - start_ms) * padding_multiplier

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
                 title=f"Series [{timeframe.upper()}] - {pd.to_datetime(start_ms, unit='ms').strftime('%Y-%m-%d %H:%M')}",
                 alines=dict(alines=lines, colors=colors, linestyle=styles, linewidths=widths),
                 addplot=addplots if addplots else None,
                 savefig=dict(fname=output_filename, dpi=150))
        print(f"성공! 단일 시리즈 차트가 '{output_filename}'에 저장되었습니다.")
    except Exception as e:
        print(f"단일 시리즈 차트 시각화 중 오류 발생: {e}")


# ==============================================================================
# ## 섹션 2: Dash 앱 레이아웃 및 컴포넌트 (수정됨)
# ==============================================================================
app = dash.Dash(__name__)
app.title = "Professional Pattern Analyzer - Dashboard"

app.layout = html.Div(style={'backgroundColor': '#1E1E1E', 'color': '#E0E0E0', 'fontFamily': 'sans-serif'}, children=[
    dcc.Store(id='analysis-data-store'),
    html.H1(children='프로페셔널 인터랙티브 패턴 분석기', style={'textAlign': 'center', 'padding': '15px'}),
    
    # --- 컨트롤 패널 ---
    html.Div([
        # ❗️❗️❗️ 파일 경로 입력 대신 드롭다운으로 변경 ❗️❗️❗️
        html.Div([
            html.H4("1. 데이터 로드", style={'marginTop': '0'}),
            dcc.Dropdown(
                id='filepath-dropdown',
                options=[
                    {'label': '1분봉 분석 결과 (1m_analysis_results_5years_robust.parquet)', 'value': '1m_analysis_results_5years_robust.parquet'},
                    {'label': '5분봉 분석 결과 (analysis_results_5years_robust.parquet)', 'value': 'analysis_results_5years_robust.parquet'},
                    {'label': '15분봉 분석 결과 (15m_analysis_results_5years_robust.parquet)', 'value': '15m_analysis_results_5years_robust.parquet'},
                ],
                value='analysis_results_5years_robust.parquet', # 기본 선택값
                style={'width': '100%', 'color': '#1E1E1E'},
                clearable=False
            ),
            html.Button('데이터 불러오기', id='load-data-button', n_clicks=0, style={'padding': '8px 15px', 'marginLeft': '20px', 'marginTop': '10px'}),
        ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px', 'marginBottom': '10px'}),
        
        html.Div([
            html.H4("2. 필터 조건 설정", style={'marginTop': '0'}),
            dcc.Input(id='lookaround-input', type='number', value=5, style={'display': 'none'}),
            dcc.Input(id='tolerance-input', type='number', value=0.001, style={'display': 'none'}),
            html.Div([
                html.Div([html.Label("되돌림 점수 (최소/최대):"), dcc.Input(id='ret-score-min', type='number'), dcc.Input(id='ret-score-max', type='number')], style={'display': 'inline-block', 'padding': '5px 15px'}),
                html.Div([html.Label("피봇 개수 (최소/최대):"), dcc.Input(id='pivot-min', type='number'), dcc.Input(id='pivot-max', type='number')], style={'display': 'inline-block', 'padding': '5px 15px'}),
                html.Div([html.Label("절대 시각적 각도 (도):"), dcc.Input(id='slope-min', type='number'), dcc.Input(id='slope-max', type='number')], style={'display': 'inline-block', 'padding': '5px 15px'}),
                html.Div([html.Label("방향:"), dcc.Dropdown(id='direction-dropdown', options=[{'label': '전체', 'value': 'all'}, {'label': '상승(UP)', 'value': 'up'}, {'label': '하락(DOWN)', 'value': 'down'}], value='all', clearable=False, style={'color': '#1E1E1E'})], style={'display': 'inline-block', 'width': '200px', 'padding': '5px 15px', 'verticalAlign': 'middle'}),
            ], style={'textAlign': 'center'}),
            html.Button('필터 적용 및 그래프 갱신', id='filter-button', n_clicks=0, style={'padding': '8px 15px', 'marginTop': '10px'}),
        ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px', 'marginBottom': '10px'}),
        
        html.Div([
            html.H4("3. 수동 스크린샷 생성 (인덱스)", style={'marginTop': '0'}),
            dcc.Input(id='index-input', type='number', placeholder='분석할 원본 인덱스 번호 입력...', style={'width': '250px'}),
            html.Button('인덱스로 스크린샷 생성', id='screenshot-by-index-button', n_clicks=0, style={'padding': '8px 15px', 'marginLeft': '20px'}),
        ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px'}),
    ], style={'padding': '10px'}),

    
    dcc.Loading(id="loading-icon", children=[dcc.Graph(id='vector-3d-chart', style={'height': '70vh'})], type="default"),
    html.Div(id='click-output', style={'textAlign': 'center', 'padding': '15px', 'fontSize': '16px', 'fontWeight': 'bold'})
])

# ==============================================================================
# ## 섹션 3: Dash 콜백 (수정됨)
# ==============================================================================

@app.callback(
    Output('analysis-data-store', 'data'),
    Input('load-data-button', 'n_clicks'),
    State('filepath-dropdown', 'value'), # ❗️ State가 filepath-dropdown을 바라보도록 수정
    prevent_initial_call=True
)
def load_analysis_data(n_clicks, filepath):
    """분석 파일을 로드하고 타임프레임 정보를 함께 저장합니다."""
    if not filepath: return None
    try:
        df = pd.read_parquet(filepath)
        
        # 파일 이름에서 타임프레임 추정
        timeframe = "5m" # 기본값
        if '1m_analysis' in filepath:
            timeframe = '1m'
        elif '15m_analysis' in filepath:
            timeframe = '15m'
        
        vectors_with_indices = list(enumerate(df.values.tolist()))
        print(f"'{filepath}' ({timeframe})에서 {len(vectors_with_indices)}개의 데이터를 성공적으로 불러왔습니다.")
        # ❗️ 데이터와 함께 타임프레임도 저장
        return {'vectors': vectors_with_indices, 'timeframe': timeframe}
    except Exception as e:
        print(f"파일 로드 오류: {e}")
        return None

@app.callback(
    Output('vector-3d-chart', 'figure'),
    Output('click-output', 'children'),
    Input('filter-button', 'n_clicks'),
    Input('vector-3d-chart', 'clickData'),
    Input('screenshot-by-index-button', 'n_clicks'),
    State('analysis-data-store', 'data'),
    State('ret-score-min', 'value'), State('ret-score-max', 'value'),
    State('pivot-min', 'value'), State('pivot-max', 'value'),
    State('slope-min', 'value'), State('slope-max', 'value'),
    State('direction-dropdown', 'value'),
    State('index-input', 'value'),
    State('lookaround-input', 'value'),
    State('tolerance-input', 'value'),
    prevent_initial_call=True
)
def update_graph_and_handle_actions(
    filter_clicks, clickData, screenshot_by_index_clicks,
    analysis_data, 
    rs_min, rs_max, p_min, p_max, s_min, s_max, direction,
    target_index,
    lookaround, tolerance
):
    """❗️❗️❗️ 그래프 업데이트 및 스크린샷 생성 (타임프레임 인지 기능 추가) ❗️❗️❗️"""
    if not analysis_data or 'vectors' not in analysis_data:
        return go.Figure().update_layout(title_text="'데이터 불러오기' 버튼을 눌러 분석 결과를 로드하세요.", template='plotly_dark'), "대기 중..."

    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'
    
    # ❗️ 저장소에서 타임프레임 정보 가져오기
    timeframe = analysis_data.get('timeframe', '5m')
    vectors_with_indices = analysis_data['vectors']
    message = ""

    screenshot_request_index = None
    if triggered_id == 'vector-3d-chart' and clickData:
        screenshot_request_index = clickData['points'][0]['customdata']
    elif triggered_id == 'screenshot-by-index-button' and target_index is not None:
        screenshot_request_index = int(target_index)

    if screenshot_request_index is not None:
        original_vector_item = next((item for item in vectors_with_indices if item[0] == screenshot_request_index), None)
        
        if original_vector_item:
            original_vector = original_vector_item[1]
            start_ts, end_ts = int(original_vector[0]), int(original_vector[1])
            
            # ❗️ 타임프레임별로 데이터 로드 시의 패딩을 다르게 설정
            padding_map_fetch = {'1m': 1.5, '5m': 1.0, '15m': 0.8}
            padding_multiplier = padding_map_fetch.get(timeframe, 1.0)
            padding = (end_ts - start_ts) * padding_multiplier
            
            print(f"인덱스 {screenshot_request_index}의 상세 분석을 시작합니다 (타임프레임: {timeframe})...")
            # ❗️ fetch_klines 호출 시 타임프레임 변수 사용
            df_local = fetch_klines("BTCUSDT", timeframe, int(start_ts - padding), int(end_ts + padding))
            
            if not df_local.empty:
                local_pivots = find_pivots_optimized(df_local, lookaround)
                local_series_sequence = build_hybrid_series_sequence(df_local, local_pivots, tolerance)
                
                target_series = min(local_series_sequence, key=lambda s: abs(s['shape']['x0'] - start_ts) + abs(s['shape']['x1'] - end_ts))
                # ❗️ 파일 이름에 타임프레임 추가
                output_filename = f"screenshot_{timeframe}_{datetime.fromtimestamp(start_ts/1000).strftime('%Y%m%d_%H%M%S')}_series_{screenshot_request_index}.png"
                
                # ❗️ 시각화 함수에 타임프레임 전달
                visualize_single_series_and_save(df_local, local_series_sequence, target_series, output_filename, local_pivots, timeframe)
                message = f"✅ 스크린샷 저장 완료: {output_filename}"
            else:
                message = f"❌ 스크린샷 생성 실패 (인덱스 {screenshot_request_index}): 해당 구간의 데이터를 가져올 수 없습니다."
        else:
            message = f"❌ 스크린샷 생성 실패: 원본 인덱스 {screenshot_request_index}를 찾을 수 없습니다."

    # --- 필터링 로직 (항상 실행하여 그래프를 최신 상태로 유지) ---
    filtered_vectors_with_indices = [
        item for item in vectors_with_indices
        if (rs_min is None or item[1][2] >= rs_min) and
           (rs_max is None or item[1][2] <= rs_max) and
           (p_min is None or item[1][3] >= p_min) and
           (p_max is None or item[1][3] <= p_max) and
           (s_min is None or item[1][4] >= s_min) and
           (s_max is None or item[1][4] <= s_max) and
           (direction == 'all' or (direction == 'up' and item[1][5] == 1.0) or (direction == 'down' and item[1][5] == -1.0))
    ]

    graph_message = f"필터링된 데이터: {len(filtered_vectors_with_indices)}개"
    final_message = message if message else graph_message

    # --- 3D 플롯 생성 로직 ---
    if not filtered_vectors_with_indices:
        return go.Figure().update_layout(title_text='필터 조건에 맞는 데이터가 없습니다.', template='plotly_dark'), final_message

    original_indices, filtered_vectors = zip(*filtered_vectors_with_indices)

    hover_texts = [
        (f"원본 인덱스: {idx}<br>"
         f"시작: {datetime.fromtimestamp(v[0]/1000).strftime('%y/%m/%d %H:%M')}<br>"
         f"종료: {datetime.fromtimestamp(v[1]/1000).strftime('%y/%m/%d %H:%M')}<br>"
         f"--------------------<br>"
         f"되돌림 점수: {v[2]:.2f}<br>"
         f"피봇 개수: {v[3]}<br>"
         f"절대 각도: {v[4]:.2f}°<br>"
         f"방향: {'UP' if v[5] == 1.0 else 'DOWN'}")
        for idx, v in zip(original_indices, filtered_vectors)
    ]

    fig = go.Figure(data=[go.Scatter3d(
        x=[v[2] for v in filtered_vectors],
        y=[v[4] for v in filtered_vectors],
        z=[v[3] for v in filtered_vectors],
        mode='markers',
        customdata=original_indices,
        marker=dict(size=5, symbol='square', color=['lightgreen' if v[5] == 1.0 else 'lightcoral' for v in filtered_vectors], opacity=0.8),
        text=hover_texts, hoverinfo='text'
    )])
    
    fig.update_layout(
        title_text=f'3D 벡터 공간 (총 {len(vectors_with_indices)}개 중 {len(filtered_vectors_with_indices)}개 표시)',
        template='plotly_dark',
        scene=dict(xaxis_title='되돌림 점수', yaxis_title='절대 시각적 각도 (도)', zaxis_title='피봇 개수'),
        margin=dict(l=10, r=20, b=10, t=50)
    )

    return fig, final_message

# ==============================================================================
# ## 섹션 4: 애플리케이션 실행
# ==============================================================================
if __name__ == '__main__':
    print("\n### 사용 안내 ###")
    print("1. 데이터 서버(server_5min.cjs)를 실행하세요.")
    print("2. 분석 결과(.parquet) 파일들이 스크립트와 같은 폴더에 있는지 확인하세요.")
    print("3. 이 대시보드 스크립트를 실행하고 웹 브라우저에서 http://127.0.0.1:8050 주소로 접속하세요.")
    print("4. 드롭다운에서 분석할 타임프레임 선택 -> '데이터 불러오기' -> (선택)필터 적용 -> 점 클릭 또는 인덱스 입력으로 스크린샷 생성.")
   
    app.run(debug=True, port=8050)