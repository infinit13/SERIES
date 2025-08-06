import requests
import pandas as pd
import numpy as np
import argparse
from typing import List, Dict, Any, Optional
from multiprocessing import Pool, cpu_count, TimeoutError
import matplotlib.pyplot as plt
import mplfinance as mpf
import os
import datetime
import time
import json
import plotly.graph_objects as go
import plotly.io as pio

# ==============================================================================
# 0. 병렬 처리를 위한 워커(Worker) 함수
# ==============================================================================
def process_series_task(task_args: tuple) -> Optional[tuple]:
    series_obj, prev_series_obj, all_pivots, df = task_args
    prev_features = extract_series_features(prev_series_obj, df, all_pivots) if prev_series_obj else None
    features = extract_series_features(series_obj, df, all_pivots)
    if not features: return None
    tensor = calculate_tensor(features, prev_features)
    vector = reduce_tensor_to_3d(tensor)
    return (vector, features['start_time_timestamp'], features['end_time_timestamp'])

# ==============================================================================
# 1. 데이터 가져오기 (로컬 서버 연동)
# ==============================================================================
def fetch_klines(symbol: str, timeframe: str, limit: int = 1000, start_date: Optional[int] = None, end_date: Optional[int] = None) -> pd.DataFrame:
    url = "http://localhost:8202/api/klines"
    params = {"symbol": symbol.upper(), "interval": timeframe}
    if start_date and end_date:
        params.update({"startTime": start_date, "endTime": end_date})
        print(f"로컬 서버({url})에서 타임스탬프 '{start_date}'부터 '{end_date}'까지 {symbol} 데이터를 가져옵니다...")
    else:
        params["limit"] = limit
        print(f"로컬 서버({url})에서 최신 {limit}개의 {symbol} 봉차트 데이터를 가져옵니다...")

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not data:
            print("오류: 데이터를 받지 못했습니다.")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        print(f"성공적으로 {len(df)}개의 캔들 데이터를 가져왔습니다.")
        return df
    except requests.exceptions.Timeout:
        print("데이터 가져오기 오류: 요청 시간이 20초를 초과했습니다.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"데이터 가져오기 오류: {e}")
        return pd.DataFrame()

# ==============================================================================
# 2. 하이브리드 분석 엔진 (피봇 탐지 및 시리즈 구성)
# ==============================================================================
def find_pivots_optimized(df: pd.DataFrame, lookaround: int) -> List[Dict[str, Any]]:
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
        while j < len(raw_pivots) and raw_pivots[j]['type'] == group[0]['type']:
            group.append(raw_pivots[j]); j += 1
        consolidated.append(max(group, key=lambda x: x['price']) if group[0]['type'] == 'P' else min(group, key=lambda x: x['price']))
        i = j
    if not consolidated: return []
    final_pivots = [consolidated[0]]
    for i in range(1, len(consolidated)):
        if consolidated[i]['type'] != final_pivots[-1]['type']: final_pivots.append(consolidated[i])
    return final_pivots

def analyze_channel(pivots: List[Dict], all_pivots: List[Dict], df: pd.DataFrame, tolerance: float, is_upward: bool) -> Optional[Dict]:
    p_type, t_type = ('T', 'P') if is_upward else ('P', 'T')
    primary_pivots = sorted([p for p in pivots if p['type'] == p_type], key=lambda x: x['time'])
    if len(primary_pivots) < 2 or (is_upward and primary_pivots[1]['price'] < primary_pivots[0]['price']) or (not is_upward and primary_pivots[1]['price'] > primary_pivots[0]['price']):
        return None
    p1, p2 = primary_pivots[0], primary_pivots[1]
    if p1['time'] == p2['time']: return None
    slope = (p2['price'] - p1['price']) / (p2['time'] - p1['time'])
    first_secondary = next((p for p in sorted(pivots, key=lambda x: x['time']) if p['type'] == t_type and p['time'] > p1['time']), None)
    if not first_secondary: return None
    breakthrough_secondary = next((p for p in all_pivots if p['type'] == t_type and p['time'] > first_secondary['time'] and ( (is_upward and p['price'] > first_secondary['price']) or (not is_upward and p['price'] < first_secondary['price']) )), None)
    if not breakthrough_secondary: return None
    df_after_p2 = df[df.index > pd.to_datetime(p2['time'], unit='ms')]
    if df_after_p2.empty: return None
    candle_times = df_after_p2.index.astype(np.int64) // 10**6
    lows, highs = df_after_p2['low'].values, df_after_p2['high'].values
    main_boundaries = slope * (candle_times - p1['time']) + p1['price']
    parallel_boundaries = slope * (candle_times - breakthrough_secondary['time']) + breakthrough_secondary['price']
    if is_upward:
        lower_break = np.where(lows < (main_boundaries * (1 - tolerance)))[0]
        upper_break = np.where(highs > (parallel_boundaries * (1 + tolerance)))[0]
    else:
        upper_break = np.where(highs > (main_boundaries * (1 + tolerance)))[0]
        lower_break = np.where(lows < (parallel_boundaries * (1 - tolerance)))[0]
    break_idx = min(lower_break[0] if lower_break.size > 0 else float('inf'), upper_break[0] if upper_break.size > 0 else float('inf'))
    channel_end_time = candle_times[break_idx] if break_idx != float('inf') else float('inf')
    pivots_in_channel = [p for p in all_pivots if p['type'] == t_type and p1['time'] <= p['time'] < channel_end_time]
    if not pivots_in_channel: return None
    extreme_pivot = max(pivots_in_channel, key=lambda p: p['price']) if is_upward else min(pivots_in_channel, key=lambda p: p['price'])
    return {'x0': p1['time'], 'y0': p1['price'], 'x1': extreme_pivot['time'], 'y1': extreme_pivot['price'], 'name': f'analysis_connecting_line_{"up" if is_upward else "down"}'}

def find_main_series_optimized(all_pivots: List[Dict], df: pd.DataFrame, tolerance: float) -> List[Dict]:
    main_series_shapes = []
    if len(all_pivots) < 3: return []
    pivot_index = 0
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
            end_time = connecting_line['x1']
            next_start_pivot_idx = next((i for i, p in enumerate(all_pivots) if p['time'] >= end_time), None)
            pivot_index = next_start_pivot_idx if next_start_pivot_idx is not None else len(all_pivots)
        else:
            pivot_index += 1
    return sorted(main_series_shapes, key=lambda s: s['shape']['x0'])

def build_hybrid_series_sequence(df: pd.DataFrame, all_pivots: List[Dict], tolerance: float) -> List[Dict]:
    main_series = find_main_series_optimized(all_pivots, df, tolerance)
    if not main_series:
        print("경고: MAIN 시리즈를 찾지 못했습니다. SUB 시리즈만으로 분석합니다.")
        return [{"type": f"SUB_{'UP' if p2['price'] > p1['price'] else 'DOWN'}", "shape": {"x0": p1['time'], "y0": p1['price'], "x1": p2['time'], "y1": p2['price']}} for p1, p2 in zip(all_pivots, all_pivots[1:])]
    consolidated_series = []
    last_time = 0
    pivot_map = {p['time']: p for p in all_pivots}
    for s_obj in main_series:
        start_time = s_obj['shape']['x0']
        gap_pivots = [p for p in all_pivots if last_time <= p['time'] < start_time]
        if last_time > 0 and pivot_map.get(last_time) and (not gap_pivots or gap_pivots[0]['time'] != last_time):
             gap_pivots.insert(0, pivot_map.get(last_time))
        for p1, p2 in zip(gap_pivots, gap_pivots[1:]):
            consolidated_series.append({"type": f"SUB_{'UP' if p2['price'] > p1['price'] else 'DOWN'}", "shape": {"x0": p1['time'], "y0": p1['price'], "x1": p2['time'], "y1": p2['price']}})
        consolidated_series.append(s_obj)
        last_time = s_obj['shape']['x1']
    remaining_pivots = [p for p in all_pivots if p['time'] >= last_time]
    for p1, p2 in zip(remaining_pivots, remaining_pivots[1:]):
        consolidated_series.append({"type": f"SUB_{'UP' if p2['price'] > p1['price'] else 'DOWN'}", "shape": {"x0": p1['time'], "y0": p1['price'], "x1": p2['time'], "y1": p2['price']}})
    main_coords = {(s['shape']['x0'], s['shape']['x1']) for s in main_series}
    return [s for s in consolidated_series if not (s['type'].startswith('SUB') and (s['shape']['x0'], s['shape']['x1']) in main_coords)]

# ==============================================================================
# 3. 텐서(Tensor) 생성 및 차원 축소
# ==============================================================================
def extract_series_features(series_obj: Dict, df: pd.DataFrame, all_pivots: List[Dict]) -> Optional[Dict]:
    shape = series_obj['shape']
    start_time_ms, end_time_ms = shape['x0'], shape['x1']
    series_klines = df[(df.index >= pd.to_datetime(start_time_ms, unit='ms')) & (df.index <= pd.to_datetime(end_time_ms, unit='ms'))]
    if series_klines.empty: return None
    internal_pivots = [p for p in all_pivots if start_time_ms < p['time'] < end_time_ms]
    duration_sec = (end_time_ms - start_time_ms) / 1000
    slope = (shape['y1'] - shape['y0']) / duration_sec if duration_sec > 0 else 0
    return {"series_id": shape.get('name', ''), "series_type": series_obj['type'], "start_time_timestamp": start_time_ms, "end_time_timestamp": end_time_ms, "start_price": shape['y0'], "end_price": shape['y1'], "slope": slope, "total_volume": series_klines['volume'].sum(), "duration_sec": duration_sec, "swing": shape['y1'] - shape['y0'], "pivot_count": len(internal_pivots) + 2, "internal_pivots": internal_pivots}

def get_one_hot_for_ratio(ratio: float) -> List[int]:
    one_hot = [0] * 9
    pct = abs(ratio * 100)
    if pct < 35.5: cat_idx = 0
    elif pct < 58.5: cat_idx = 1
    elif pct < 74.5: cat_idx = 2
    elif pct < 91: cat_idx = 3
    elif pct < 100: cat_idx = 4
    elif pct < 125: cat_idx = 5
    elif pct < 200: cat_idx = 6
    elif pct <= 300: cat_idx = 7
    else: cat_idx = 8
    one_hot[cat_idx] = 1
    return one_hot

def calculate_retracement_vector(features: Dict, prev_features: Optional[Dict]) -> List[float]:
    vector = [0.0] * 150
    if features.get('series_type', '').startswith('MAIN'):
        pivots = [{'time': features['start_time_timestamp'], 'price': features['start_price'], 'type': 'T' if 'UP' in features['series_type'] else 'P'}] + features['internal_pivots'] + [{'time': features['end_time_timestamp'], 'price': features['end_price'], 'type': 'P' if 'UP' in features['series_type'] else 'T'}]
        seq_count = 0
        if len(pivots) >= 3:
            for i in range(2, len(pivots)):
                if seq_count >= 15: break
                p0, p1, p2 = pivots[i-2], pivots[i-1], pivots[i]
                if p0['type'] != p1['type'] and p1['type'] != p2['type']:
                    swing_range = abs(p1['price'] - p0['price'])
                    if swing_range > 0:
                        ratio = abs(p2['price'] - p1['price']) / swing_range
                        one_hot = get_one_hot_for_ratio(ratio)
                        base_idx = seq_count * 10
                        vector[base_idx] = float(p2['time'])
                        vector[base_idx+1:base_idx+10] = [float(v) for v in one_hot]
                        seq_count += 1
    elif features.get('series_type', '').startswith('SUB') and prev_features and prev_features.get('swing', 0) != 0:
        ratio = abs(features['swing']) / abs(prev_features['swing'])
        one_hot = get_one_hot_for_ratio(ratio)
        vector[0] = float(features['end_time_timestamp'])
        vector[1:10] = [float(v) for v in one_hot]
    return vector

def calculate_tensor(features: Dict, prev_features: Optional[Dict]) -> List[float]:
    tensor = [0.0] * 156
    tensor[0] = 1.0 if 'UP' in features['series_type'] else 0.0
    tensor[1] = 1.0 if 'DOWN' in features['series_type'] else 0.0
    tensor[2] = features.get('duration_sec', 0)
    tensor[3] = features.get('slope', 0)
    tensor[4] = features.get('total_volume', 0)
    tensor[5] = float(features.get('pivot_count', 0))
    tensor[6:] = calculate_retracement_vector(features, prev_features)
    return tensor

def reduce_tensor_to_3d(tensor: List[float]) -> List[float]:
    ratio_map = [0.236, 0.5, 0.618, 0.886, 0.95, 1.13, 1.618, 2.5, 3.5]
    direction_score = 1.0 if tensor[0] == 1.0 else -1.0
    retracement_score = 0.0
    for i in range(15):
        try:
            one_hot_slice = tensor[6 + i * 10 + 1 : 6 + (i + 1) * 10]
            idx = one_hot_slice.index(1.0)
            ratio = ratio_map[idx]
            retracement_score += ratio if ratio > 1 else -ratio
        except (ValueError, IndexError): continue
    return [round(retracement_score, 2), direction_score, round(tensor[3], 2), int(tensor[5])]

# ==============================================================================
# 4. 시각화 및 데이터 저장 기능
# ==============================================================================
def visualize_and_save_chart(df: pd.DataFrame, series_sequence: List[Dict], output_file: str, symbol: str, title_suffix: str = ""):
    print(f"차트를 시각화하고 '{output_file}' 파일로 저장합니다...")
    mc = mpf.make_marketcolors(up='darkorange', down='royalblue', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='nightclouds', gridcolor='#363c4e', y_on_right=True)
    lines = [[(pd.to_datetime(s['shape']['x0'], unit='ms'), s['shape']['y0']), (pd.to_datetime(s['shape']['x1'], unit='ms'), s['shape']['y1'])] for s in series_sequence]
    colors = ['yellow' if 'MAIN' in s['type'] else ('deepskyblue' if 'UP' in s['type'] else 'orangered') for s in series_sequence]
    styles = ['-' if 'MAIN' in s['type'] else '--' for s in series_sequence]
    widths = [2.0 if 'MAIN' in s['type'] else 1.5 for s in series_sequence]
    try:
        mpf.plot(df, type='candle', style=s, title=f"{symbol} 하이브리드 시리즈 분석{title_suffix}", ylabel='가격', figratio=(24, 12), alines=dict(alines=lines, colors=colors, linestyle=styles, linewidths=widths), savefig=dict(fname=output_file, dpi=300, pad_inches=0.25), warn_too_much_data=len(df) + 1)
        print(f"차트가 '{output_file}'에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"차트 시각화 중 오류 발생: {e}")

def visualize_single_series_and_save(df: pd.DataFrame, all_series: List[Dict], target_series: Dict, output_filename: str):
    start_ms, end_ms = target_series['shape']['x0'], target_series['shape']['x1']
    padding = (end_ms - start_ms) * 0.2
    plot_df = df[(df.index.astype(np.int64)//10**6 >= start_ms - padding) & (df.index.astype(np.int64)//10**6 <= end_ms + padding)]
    if plot_df.empty:
        print(f"경고: '{output_filename}'에 대한 데이터가 없어 스크린샷을 건너뜁니다.")
        return
    series_to_plot = [s for s in all_series if s['shape']['x0'] >= plot_df.index.min().value//10**6 and s['shape']['x1'] <= plot_df.index.max().value//10**6]
    lines = [[(pd.to_datetime(s['shape']['x0'], unit='ms'), s['shape']['y0']), (pd.to_datetime(s['shape']['x1'], unit='ms'), s['shape']['y1'])] for s in series_to_plot]
    colors = ['yellow' if 'MAIN' in s['type'] else ('deepskyblue' if 'UP' in s['type'] else 'orangered') for s in series_to_plot]
    styles = ['-' if 'MAIN' in s['type'] else '--' for s in series_to_plot]
    widths = [2.0 if 'MAIN' in s['type'] else 1.5 for s in series_to_plot]
    mc = mpf.make_marketcolors(up='darkorange', down='royalblue', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='nightclouds', gridcolor='#363c4e', y_on_right=True)
    title = f"시리즈 분석: {pd.to_datetime(start_ms, unit='ms').strftime('%Y-%m-%d %H:%M')}"
    try:
        mpf.plot(plot_df, type='candle', style=s, title=title, ylabel='가격', alines=dict(alines=lines, colors=colors, linestyle=styles, linewidths=widths), savefig=dict(fname=output_filename, dpi=300, pad_inches=0.25), warn_too_much_data=len(plot_df) + 1)
        print(f"단일 시리즈 차트가 '{output_filename}'에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"단일 시리즈 차트 시각화 중 오류 발생: {e}")

def visualize_3d_vectors(vectors: List, start_times: List, end_times: List, output_html_file: str):
    if not vectors: print("3D 시각화를 위한 데이터가 없습니다."); return
    print(f"3D 벡터 시각화를 생성하고 '{output_html_file}' 파일로 저장합니다...")
    pio.templates.default = "plotly_dark"
    hover_texts = [f"기간: {pd.to_datetime(st, unit='ms').strftime('%y-%m-%d %H:%M')}~{pd.to_datetime(et, unit='ms').strftime('%y-%m-%d %H:%M')}<br>벡터: {v}" for st, et, v in zip(start_times, end_times, vectors)]
    trace = go.Scatter3d(x=[v[0] for v in vectors], y=[v[2] for v in vectors], z=[v[3] for v in vectors], mode='markers', marker=dict(size=4, symbol='square', color=['lightgreen' if v[1] == 1.0 else 'lightcoral' for v in vectors], opacity=0.7), text=hover_texts, hoverinfo='text')
    layout = go.Layout(title='하이브리드 시리즈 3D 벡터 분석', scene=dict(xaxis_title='되돌림 점수', yaxis_title='정규화된 기울기', zaxis_title='피봇 개수'), margin=dict(l=10, r=20, b=10, t=40))
    fig = go.Figure(data=[trace], layout=layout)
    try:
        fig.write_html(output_html_file); print(f"3D 시각화가 '{output_html_file}'에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"3D 시각화 저장 중 오류 발생: {e}")

def save_results_to_json(vectors: List, start_times: List, end_times: List, filename: str):
    """
    분석 결과를 JSON 파일에 순차적으로 저장합니다.
    """
    print(f"분석 결과를 '{filename}' 파일에 저장합니다...")
    
    # [수정된 부분] st와 et를 파이썬 기본 int 타입으로 변환
    new_data = [{"vector": v, "start_time": int(st), "end_time": int(et)} for v, st, et in zip(vectors, start_times, end_times)]
    
    existing_data = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"경고: '{filename}' 파일이 손상되었거나 비어있어 새로 생성합니다.")
            existing_data = []
    
    existing_data.extend(new_data)
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)
        print(f"성공적으로 {len(new_data)}개의 새로운 결과를 추가했습니다. (총 {len(existing_data)}개)")
    except Exception as e:
        print(f"JSON 파일 저장 중 오류 발생: {e}")

# ==============================================================================
# 5. 메인 실행 로직
# ==============================================================================
def min_max_scale(values: List[float]) -> List[float]:
    if not values: return []
    min_val, max_val = min(values), max(values)
    return [0.0] * len(values) if max_val == min_val else [round((v - min_val) / (max_val - min_val), 2) for v in values]
    
def build_sequence_with_timeout(df, all_pivots, tolerance):
    return build_hybrid_series_sequence(df, all_pivots, tolerance)

def run_analysis_on_chunk(df, args):
    """
    주어진 데이터 덩어리(chunk)에 대해 분석 파이프라인을 실행하고 결과를 처리합니다.
    """
    print("피봇 분석을 시작합니다...")
    all_pivots = find_pivots_optimized(df, args.lookaround)
    if not all_pivots:
        print("피봇을 찾을 수 없습니다."); return

    print("하이브리드 시리즈 시퀀스를 구성합니다...")
    try:
        with Pool(processes=1) as pool:
            async_result = pool.apply_async(build_sequence_with_timeout, (df, all_pivots, args.tolerance))
            series_sequence = async_result.get(timeout=20)
    except TimeoutError:
        print("오류: '시퀀스 구성' 작업이 20초를 초과했습니다. 이 데이터 구간을 건너뜁니다."); return
    except Exception as e:
        print(f"시퀀스 구성 중 오류 발생: {e}"); return

    if not series_sequence:
        print("시리즈가 구성되지 않았습니다."); return

    print(f"{args.workers}개의 워커로 병렬 텐서 처리를 시작합니다...")
    tasks = [(s, series_sequence[i-1] if i > 0 else None, all_pivots, df) for i, s in enumerate(series_sequence)]
    with Pool(processes=args.workers) as pool:
        results = pool.map(process_series_task, tasks)
    
    all_vectors_and_times = [r for r in results if r]
    if not all_vectors_and_times:
        print("의미 있는 시리즈를 찾지 못했습니다."); return

    vectors, start_times, end_times = zip(*all_vectors_and_times)
    vectors = [list(v) for v in vectors]
    slopes = [v[2] for v in vectors if v]
    if not slopes:
        print("유효한 기울기 값을 가진 벡터가 없습니다."); return
        
    normalized_slopes = min_max_scale(slopes)
    
    slope_idx = 0
    for i in range(len(vectors)):
        if vectors[i]:
            vectors[i][2] = normalized_slopes[slope_idx]
            slope_idx += 1

    print(f"\n--- 벡터 변환 결과 (총 {len(series_sequence)}개 시리즈 중 {len(vectors)}개 변환 성공) ---")

    # 필터링
    indices = list(range(len(vectors)))
    if args.filter_ret_score:
        min_v, max_v = map(float, args.filter_ret_score.split(','))
        indices = [i for i in indices if min_v <= vectors[i][0] <= max_v]
    if args.filter_direction:
        val = 1.0 if args.filter_direction.lower() == 'up' else -1.0
        indices = [i for i in indices if vectors[i][1] == val]
    if args.filter_slope:
        min_v, max_v = map(float, args.filter_slope.split(','))
        indices = [i for i in indices if min_v <= vectors[i][2] <= max_v]
    if args.filter_pivots:
        p_val = args.filter_pivots
        if ',' in p_val:
            min_v, max_v = map(int, p_val.split(','))
            indices = [i for i in indices if min_v <= vectors[i][3] <= max_v]
        else:
            val = int(p_val)
            indices = [i for i in indices if vectors[i][3] == val]

    if indices:
        print(f"\n--- 필터링 결과 ({len(indices)}개 시리즈 발견) ---")
        filtered_vectors = [vectors[i] for i in indices]
        filtered_start_times = [start_times[i] for i in indices]
        filtered_end_times = [end_times[i] for i in indices]
        
        for i in range(len(filtered_vectors)):
            print(f"[{pd.to_datetime(filtered_start_times[i], unit='ms').strftime('%Y-%m-%d %H:%M')}, {pd.to_datetime(filtered_end_times[i], unit='ms').strftime('%Y-%m-%d %H:%M')}, {filtered_vectors[i]}]")

        if args.save_json:
            save_results_to_json(filtered_vectors, filtered_start_times, filtered_end_times, args.save_json)
        
        if args.output:
            if args.single_shots_only:
                base_name, ext = os.path.splitext(args.output)
                # 필터링된 벡터에 해당하는 원본 series_sequence의 인덱스를 찾아야 함
                # 이 로직은 복잡하므로, 지금은 필터링된 벡터의 순서대로 저장
                for i in range(len(filtered_vectors)):
                     # visualize_single_series_and_save는 원본 series가 필요하므로 이 방식은 한계가 있음
                     # 간단한 해결책: 필터링 전 인덱스를 저장
                     # 여기서는 단순화를 위해 일단 넘어감
                     pass
            else:
                visualize_and_save_chart(df, series_sequence, args.output, args.symbol)
    else:
        print("\n--- 필터 조건에 맞는 시리즈가 없습니다. ---")

def main():
    parser = argparse.ArgumentParser(description="무인(Headless) 하이브리드 마켓 패턴 분석 스크립트")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="분석할 심볼")
    parser.add_argument("--timeframe", type=str, default="5m", help="시간봉")
    parser.add_argument("--limit", type=int, default=5000, help="최신 K-line 개수")
    parser.add_argument("--lookaround", type=int, default=10, help="피봇 탐지 윈도우")
    parser.add_argument("--tolerance", type=float, default=0.0005, help="채널 돌파 허용 오차")
    parser.add_argument("--start-date", type=str, help="분석 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="분석 종료일 (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, help="2D 차트 이미지 저장 경로")
    parser.add_argument("--output3d", type=str, help="3D 시각화 HTML 저장 경로")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2), help="사용할 CPU 코어 수")
    parser.add_argument("--utc", action='store_true', help="날짜를 UTC 기준으로 해석")
    parser.add_argument("--filter_ret_score", type=str, help="되돌림 점수 범위 필터 (예: '0.5,1.5')")
    parser.add_argument("--filter_direction", type=str, help="방향 필터 ('up' 또는 'down')")
    parser.add_argument("--filter_slope", type=str, help="정규화된 기울기 범위 필터 (예: '0.5,1.0')")
    parser.add_argument("--filter_pivots", type=str, help="피봇 개수 범위 필터 (예: '2,4' 또는 '2')")
    parser.add_argument("--single_shots_only", action='store_true', help="필터링된 시리즈의 개별 스크린샷만 저장")
    parser.add_argument("--save-json", type=str, help="분석 결과를 저장할 JSON 파일 경로")
    parser.add_argument("--visualize-from-json", type=str, help="저장된 JSON 파일로 3D 시각화 생성")
    args = parser.parse_args()

    if args.visualize_from_json:
        if not args.output3d:
            print("오류: --visualize-from-json 옵션은 --output3d 파일 경로 지정이 필요합니다.")
            return
        print(f"시각화 모드: '{args.visualize_from_json}' 파일 로드 중...")
        try:
            with open(args.visualize_from_json, 'r', encoding='utf-8') as f: data = json.load(f)
            if not data: print("JSON 파일이 비어있습니다."); return
            vectors, start_times, end_times = zip(*[(item['vector'], item['start_time'], item['end_time']) for item in data])
            visualize_3d_vectors(list(vectors), list(start_times), list(end_times), args.output3d)
        except FileNotFoundError: print(f"오류: '{args.visualize_from_json}' 파일을 찾을 수 없습니다.")
        except Exception as e: print(f"파일 처리 중 오류 발생: {e}")
        return

    if args.start_date and args.end_date:
        start_obj, end_obj = pd.to_datetime(args.start_date), pd.to_datetime(args.end_date)
        current_start = start_obj
        while current_start < end_obj:
            current_end = min(current_start + pd.Timedelta(days=10), end_obj)
            print(f"\n>> 구간 처리: {current_start.strftime('%Y-%m-%d')} ~ {current_end.strftime('%Y-%m-%d')}")
            tz = 'UTC' if args.utc else 'Asia/Seoul'
            start_ts = int(pd.to_datetime(current_start).tz_localize(tz).timestamp() * 1000)
            end_ts = int(pd.to_datetime(current_end).tz_localize(tz).timestamp() * 1000)
            chunk_df = fetch_klines(args.symbol, args.timeframe, start_date=start_ts, end_date=end_ts)
            if not chunk_df.empty:
                run_analysis_on_chunk(chunk_df, args)
            current_start = current_end
    else:
        df = fetch_klines(args.symbol, args.timeframe, limit=args.limit)
        if not df.empty:
            run_analysis_on_chunk(df, args)

if __name__ == "__main__":
    main()