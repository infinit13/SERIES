# batch_analyzer.py (Timeout and Small Chunks Version)

import pandas as pd
import numpy as np
import requests
import os
import math
from datetime import datetime
from multiprocessing import Pool, TimeoutError # TimeoutError 임포트
import pyarrow as pa
import pyarrow.parquet as pq

# ==============================================================================
# ## 섹션 1: 분석 알고리즘 (이전과 동일)
# ==============================================================================
# fetch_klines, find_pivots_optimized 등 모든 분석 함수는 이전과 동일하게 유지된다.
# ... (이전 코드의 모든 분석 함수를 여기에 붙여넣기) ...
def fetch_klines(symbol: str, timeframe: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    url = "http://localhost:8202/api/klines"
    params = {"symbol": symbol.upper(), "interval": timeframe, "startTime": start_ts, "endTime": end_ts}
    print(f"데이터 서버({url})에서 타임스탬프 '{start_ts}'부터 '{end_ts}'까지 데이터를 요청합니다...")
    try:
        response = requests.get(url, params=params, timeout=300)
        response.raise_for_status()
        data = response.json()
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        print(f"성공적으로 {len(df)}개의 캔들 데이터를 가져왔습니다.")
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

def extract_series_features(series_obj, df, all_pivots):
    shape = series_obj['shape']
    start_time_ms, end_time_ms = shape['x0'], shape['x1']
    series_klines = df[(df.index >= pd.to_datetime(start_time_ms, unit='ms')) & (df.index <= pd.to_datetime(end_time_ms, unit='ms'))]
    if series_klines.empty: return None
    internal_pivots = [p for p in all_pivots if start_time_ms < p['time'] < end_time_ms]
    duration_sec = (end_time_ms - start_time_ms) / 1000
    slope = (shape['y1'] - shape['y0']) / duration_sec if duration_sec > 0 else 0
    return {"series_type": series_obj['type'], "start_time_timestamp": start_time_ms, "end_time_timestamp": end_time_ms, "start_price": shape['y0'], "end_price": shape['y1'], "slope": slope, "total_volume": series_klines['volume'].sum(), "duration_sec": duration_sec, "swing": shape['y1'] - shape['y0'], "pivot_count": len(internal_pivots) + 2, "internal_pivots": internal_pivots}

def calculate_visual_angle(series_features, df):
    start_ms, end_ms, raw_slope = series_features['start_time_timestamp'], series_features['end_time_timestamp'], series_features['slope']
    padding = (end_ms - start_ms) * 0.5
    local_df = df[(df.index.astype(np.int64)//10**6 >= start_ms - padding) & (df.index.astype(np.int64)//10**6 <= end_ms + padding)]
    if local_df.empty: return 0
    local_price_range = local_df['high'].max() - local_df['low'].min()
    local_time_span_sec = (local_df.index.max() - local_df.index.min()).total_seconds()
    if local_time_span_sec == 0 or local_price_range == 0: return 0
    local_price_time_ratio = local_price_range / local_time_span_sec
    normalized_slope = raw_slope / local_price_time_ratio
    angle_rad = math.atan(normalized_slope)
    angle_deg = math.degrees(angle_rad)
    return round(abs(angle_deg), 2)

def get_one_hot_for_ratio(ratio):
    one_hot, pct = [0] * 9, abs(ratio * 100)
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

def calculate_retracement_vector(features, prev_features):
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
                        vector[base_idx:base_idx+10] = [float(p2['time'])] + [float(v) for v in one_hot]
                        seq_count += 1
    elif features.get('series_type', '').startswith('SUB') and prev_features and prev_features.get('swing', 0) != 0:
        ratio = abs(features['swing']) / abs(prev_features['swing'])
        vector[0:10] = [float(features['end_time_timestamp'])] + [float(v) for v in get_one_hot_for_ratio(ratio)]
    return vector

def process_single_series(args_tuple):
    series_obj, prev_series_obj, all_pivots, df = args_tuple
    features = extract_series_features(series_obj, df, all_pivots)
    if not features: return None
    visual_angle = calculate_visual_angle(features, df)
    prev_features = extract_series_features(prev_series_obj, df, all_pivots) if prev_series_obj else None
    retracement_vec = calculate_retracement_vector(features, prev_features)
    ratio_map = [0.236, 0.5, 0.618, 0.886, 0.95, 1.13, 1.618, 2.5, 3.5]
    retracement_score = 0.0
    for i in range(15):
        try:
            idx = retracement_vec[i * 10 + 1 : (i + 1) * 10].index(1.0)
            retracement_score += ratio_map[idx] if ratio_map[idx] > 1 else -ratio_map[idx]
        except (ValueError, IndexError): continue
    return [features['start_time_timestamp'], features['end_time_timestamp'], round(retracement_score, 2), features['pivot_count'], visual_angle, 1.0 if 'UP' in features['series_type'] else -1.0]

# ==============================================================================
# ## 섹션 2: 메인 실행 로직 (분할 처리 및 타임아웃 기능 추가)
# ==============================================================================
if __name__ == '__main__':
    # --- 분석할 조건 설정 ---
    START_DATE_STR = '2019-12-01'
    END_DATE_STR = '2025-07-01'
    CHUNK_FREQUENCY = '90D' # 데이터를 10일 단위로 분할
    LOOKAROUND = 5
    TOLERANCE = 0.001
    SYMBOL = "BTCUSDT"
    TIMEFRAME = "1d"
    OUTPUT_FILENAME = "1d_analysis_results_5years_robust.parquet"
    CPU_CORES = 8
    # -------------------------

    print(f"--- 배치 분석 시작 ---")
    print(f"전체 기간: {START_DATE_STR} ~ {END_DATE_STR} ({CHUNK_FREQUENCY} 단위로 분할 처리)")
    
    date_chunks = pd.date_range(start=START_DATE_STR, end=END_DATE_STR, freq=CHUNK_FREQUENCY)
    if pd.to_datetime(END_DATE_STR) not in date_chunks:
        date_chunks = date_chunks.append(pd.Index([pd.to_datetime(END_DATE_STR)]))
    
    writer = None

    for i in range(len(date_chunks) - 1):
        chunk_start_date = date_chunks[i]
        chunk_end_date = date_chunks[i+1]
        
        chunk_start_str = chunk_start_date.strftime('%Y-%m-%d')
        chunk_end_str = chunk_end_date.strftime('%Y-%m-%d')
        start_ts = int(chunk_start_date.timestamp() * 1000)
        end_ts = int(chunk_end_date.timestamp() * 1000)

        print(f"\n--- [{i+1}/{len(date_chunks)-1}] 조각 처리 중: {chunk_start_str} ~ {chunk_end_str} ---")
        
        df = fetch_klines(SYMBOL, TIMEFRAME, start_ts, end_ts)
        if df.empty:
            print("해당 기간에 데이터가 없어 다음 조각으로 넘어갑니다.")
            continue

        pivots = find_pivots_optimized(df, LOOKAROUND)
        if not pivots:
            print("피봇을 찾을 수 없어 다음 조각으로 넘어갑니다.")
            continue
            
        series_sequence = None
        try:
            # ❗️❗️❗️ 타임아웃 로직: 1개 프로세스로 구성 작업을 시도하고 30초 이상 걸리면 건너뛴다 ❗️❗️❗️
            with Pool(processes=1) as pool:
                async_result = pool.apply_async(build_hybrid_series_sequence, (df, pivots, TOLERANCE))
                series_sequence = async_result.get(timeout=30) 
        except TimeoutError:
            print(f"경고: 시리즈 구성 작업이 30초를 초과했습니다. 이 조각({chunk_start_str})을 건너뜁니다.")
            continue
        except Exception as e:
            print(f"시리즈 구성 중 예상치 못한 오류 발생: {e}")
            continue

        if not series_sequence:
            print("시리즈가 구성되지 않아 다음 조각으로 넘어갑니다.")
            continue
        
        tasks = [(series_sequence[j], series_sequence[j-1] if j > 0 else None, pivots, df) for j in range(len(series_sequence))]
        
        print(f"{len(series_sequence)}개의 시리즈에 대한 특징 벡터 계산을 시작합니다... (CPU {CPU_CORES}개 사용)")
        
        with Pool(processes=CPU_CORES) as pool: 
            vectors = pool.map(process_single_series, tasks)
        
        vectors = [v for v in vectors if v is not None]
        
        if vectors:
            print(f"총 {len(vectors)}개의 유효한 벡터를 생성했습니다.")
            chunk_df = pd.DataFrame(vectors, columns=['start_ts', 'end_ts', 'retracement_score', 'pivot_count', 'abs_angle_deg', 'direction'])
            table = pa.Table.from_pandas(chunk_df, preserve_index=False)
            
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_FILENAME, table.schema)
            
            writer.write_table(table)
            print(f"결과가 '{OUTPUT_FILENAME}'에 추가되었습니다.")
        else:
            print("해당 조각에서 유효한 시리즈를 찾지 못했습니다.")

    if writer:
        writer.close()
        print(f"\n--- 모든 조각 처리 완료. 최종 파일 '{OUTPUT_FILENAME}'이 생성되었습니다. ---")
    else:
        print("\n분석할 데이터가 없어 작업을 종료합니다.")