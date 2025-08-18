# batch_analyzer.py (v6 - Advanced Retracement Scoring)

import pandas as pd
import numpy as np
import requests
import os
import math
from datetime import datetime
from multiprocessing import Pool, TimeoutError
import pyarrow as pa
import pyarrow.parquet as pq

# ==============================================================================
# ## 섹션 1: 분석 알고리즘 (헬퍼 함수 추가)
# ==============================================================================
# ... fetch_klines, find_pivots_optimized 등 이전 함수들은 모두 동일 ...
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
    return round(angle_deg, 2)
    
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


# --- ❗️새로운 점수 계산 헬퍼 함수 ---
def find_retracement_pattern_score(pivots_in_series, df):
    """
    시리즈 내에서 첫 충격파동(p1->p2) 이후, p2 가격을 깨기 전에
    50%~82% 되돌림 영역을 터치하는 패턴이 있는지 확인하고 1점 또는 0점을 반환한다.
    """
    if len(pivots_in_series) < 2:
        return 0

    # 시리즈 내 첫 번째 충격 파동을 p1 -> p2로 정의
    p1 = pivots_in_series[0]
    p2 = pivots_in_series[1]

    # p1, p2 타입이 다를 경우에만 유효한 충격 파동으로 간주
    if p1['type'] == p2['type']:
        return 0
    
    is_upward_impulse = (p1['type'] == 'T' and p2['type'] == 'P')
    impulse_size = abs(p2['price'] - p1['price'])
    if impulse_size == 0:
        return 0

    # 되돌림 목표 가격 구간 계산
    if is_upward_impulse:
        # 상승 파동에 대한 하락 되돌림
        zone_top = p2['price'] - impulse_size * 0.50
        zone_bottom = p2['price'] - impulse_size * 0.82
    else:
        # 하락 파동에 대한 상승 되돌림
        zone_bottom = p2['price'] + impulse_size * 0.50
        zone_top = p2['price'] + impulse_size * 0.82

    # p2 가격을 돌파하는 지점 찾기
    break_price = p2['price']
    end_scan_pivot = None
    for i in range(2, len(pivots_in_series)):
        pivot = pivots_in_series[i]
        if is_upward_impulse and pivot['price'] > break_price:
            end_scan_pivot = pivot
            break
        if not is_upward_impulse and pivot['price'] < break_price:
            end_scan_pivot = pivot
            break
    
    # 스캔할 시간 범위 설정
    start_scan_ts = p2['time']
    end_scan_ts = end_scan_pivot['time'] if end_scan_pivot else pivots_in_series[-1]['time']

    # 해당 시간 범위의 캔들 데이터 슬라이싱
    scan_df = df[(df.index.astype(np.int64)//10**6 >= start_scan_ts) & 
                 (df.index.astype(np.int64)//10**6 <= end_scan_ts)]

    if scan_df.empty:
        return 0

    # 되돌림 영역을 터치했는지 확인
    zone_touched = False
    if is_upward_impulse:
        # 상승 후 하락 되돌림이므로 low 가격이 zone에 들어왔는지 체크
        if scan_df['low'].between(zone_bottom, zone_top).any():
            zone_touched = True
    else:
        # 하락 후 상승 되돌림이므로 high 가격이 zone에 들어왔는지 체크
        if scan_df['high'].between(zone_bottom, zone_top).any():
            zone_touched = True
            
    return 1 if zone_touched else 0


# --- ❗️핵심 수정: 새로운 스코어링 로직을 호출하는 메인 함수 ---
def process_pivots_in_series(args_tuple):
    series_obj, series_id, all_pivots, df = args_tuple
    pivots_in_series = [p for p in all_pivots if series_obj['shape']['x0'] <= p['time'] <= series_obj['shape']['x1']]
    
    if len(pivots_in_series) < 2:
        return []

    # 시리즈 전체에 대한 패턴 점수 계산
    series_score = find_retracement_pattern_score(pivots_in_series, df)

    pivot_log_entries = []
    for i in range(1, len(pivots_in_series)):
        current_pivot = pivots_in_series[i]
        prev_pivot = pivots_in_series[i-1]
        
        segment_series_obj = {'shape': {'x0': prev_pivot['time'], 'x1': current_pivot['time'], 'y0': prev_pivot['price'], 'y1': current_pivot['price']}, 'type': 'SUB'}
        segment_features = extract_series_features(segment_series_obj, df, [])
        
        if not segment_features:
            continue
            
        segment_angle = calculate_visual_angle(segment_features, df)
        
        # 각 피봇 로그에 시리즈 전체의 점수를 기록
        pivot_log_entries.append({
            'series_id': series_id, 
            'pivot_index': i, 
            'timestamp': current_pivot['time'], 
            'price': current_pivot['price'], 
            'type': current_pivot['type'], 
            'segment_angle': segment_angle,
            'series_score': series_score  # 시리즈 패턴 점수 추가
        })
        
    return pivot_log_entries


# ==============================================================================
# ## 섹션 2: 메인 실행 로직 (이전과 동일)
# ==============================================================================
if __name__ == '__main__':
    START_DATE_STR = '2020-01-01'
    END_DATE_STR = '2020-10-30'
    TIMEFRAME = "15m"
    chunk_map = {'1m': '15D', '5m': '30D', '15m': '45D', '1h': '90D', '4h': '180D', '1d': '365D', '1W': '1500D', '1M': '5000D'}
    CHUNK_FREQUENCY = chunk_map.get(TIMEFRAME.upper(), '365D')
    lookaround_map = {'1m': 12, '5m': 8, '15m': 5, '1h': 5, '4h': 4, '1d': 3, '1W': 2, '1M': 2}
    LOOKAROUND = lookaround_map.get(TIMEFRAME, 5)
    CPU_CORES = os.cpu_count() or 8
    PROCESS_TIMEOUT_SECONDS = 300
    TOLERANCE = 0.001
    SYMBOL = "BTCUSDT"
    OUTPUT_FILENAME = f"{TIMEFRAME}_pattern_scored_pivots_{START_DATE_STR}_to_{END_DATE_STR}.parquet"
    
    print("="*50); print("--- 배치 분석 시작 (v6 - 되돌림 패턴 스코어링) ---"); #...
    # ... (이하 나머지 실행 코드는 이전과 동일) ...
    date_chunks = pd.date_range(start=START_DATE_STR, end=END_DATE_STR, freq=CHUNK_FREQUENCY)
    if pd.to_datetime(END_DATE_STR) not in date_chunks:
        date_chunks = date_chunks.append(pd.Index([pd.Index([pd.to_datetime(END_DATE_STR)]).asi8.item()]))
    
    writer = None

    for i in range(len(date_chunks) - 1):
        chunk_start_date, chunk_end_date = date_chunks[i], date_chunks[i+1]
        chunk_start_str = chunk_start_date.strftime('%Y-%m-%d')
        start_ts, end_ts = int(chunk_start_date.timestamp() * 1000), int(chunk_end_date.timestamp() * 1000)

        print(f"\n--- [{i+1}/{len(date_chunks)-1}] 조각 처리 중: {chunk_start_str} ~ {chunk_end_date.strftime('%Y-%m-%d')} ---")
        
        df = fetch_klines(SYMBOL, TIMEFRAME, start_ts, end_ts)
        if df.empty or len(df) <= LOOKAROUND * 2:
            print("데이터 부족으로 건너뜁니다."); continue

        pivots = find_pivots_optimized(df, LOOKAROUND)
        if not pivots:
            print("피봇을 찾지 못해 건너뜁니다."); continue
            
        series_sequence = None
        try:
            with Pool(processes=1) as pool:
                async_result = pool.apply_async(build_hybrid_series_sequence, (df, pivots, TOLERANCE))
                series_sequence = async_result.get(timeout=PROCESS_TIMEOUT_SECONDS) 
        except TimeoutError:
            print(f"경고: 시리즈 구성 타임아웃. 이 조각을 건너뜁니다."); continue
        except Exception as e:
            print(f"시리즈 구성 중 오류 발생: {e}"); continue
        
        if not series_sequence:
            print("시리즈가 구성되지 않아 건너뜁니다."); continue
        
        tasks = []
        for idx, series_obj in enumerate(series_sequence):
            series_id = f"{chunk_start_str.replace('-', '')}_{TIMEFRAME}_{idx}" 
            tasks.append((series_obj, series_id, pivots, df))

        print(f"{len(tasks)}개 시리즈에 대한 피봇별 분석을 시작합니다...")
        
        with Pool(processes=CPU_CORES) as pool: 
            list_of_pivot_logs = pool.map(process_pivots_in_series, tasks)
        
        all_pivot_logs = [log for sublist in list_of_pivot_logs for log in sublist]
        
        if all_pivot_logs:
            print(f"총 {len(all_pivot_logs)}개의 피봇 로그를 생성했습니다.")
            chunk_df = pd.DataFrame(all_pivot_logs)
            
            table = pa.Table.from_pandas(chunk_df, preserve_index=False)
            
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_FILENAME, table.schema)
            
            writer.write_table(table)
            print(f"결과가 '{OUTPUT_FILENAME}'에 추가되었습니다.")
        else:
            print("해당 조각에서 유효한 피봇 로그를 생성하지 못했습니다.")

    if writer:
        writer.close()
        print(f"\n--- 모든 조각 처리 완료. 최종 파일 '{OUTPUT_FILENAME}'이 생성되었습니다. ---")
    else:
        print("\n분석할 데이터가 없어 작업을 종료합니다.")