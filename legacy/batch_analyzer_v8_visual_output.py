# batch_analyzer_v7_pivot_scoring.py
import pandas as pd
import numpy as np
import requests
import os
import math
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq

# ==============================================================================
# ## 섹션 1: 데이터 핸들링 및 피봇 탐지 (기존과 동일)
# ==============================================================================

def fetch_klines(symbol: str, timeframe: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """로컬 데이터 서버에서 K-line 데이터를 가져온다."""
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

def find_pivots_optimized(df: pd.DataFrame, lookaround: int) -> list:
    """최적화된 방식으로 가격의 고점(P)과 저점(T) 피봇을 찾는다."""
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

# ==============================================================================
# ## 섹션 2: ❗️새로운 핵심 스코어링 알고리즘❗️
# ==============================================================================

def score_visual_first_impulse(pivots: list, EPS: float) -> tuple[dict, list]:
    """
    피봇 시퀀스를 순회하며 '충격-되돌림-확장' 패턴을 분석하고,
    각 조건에 맞는 피봇에 점수를 부여한다.
    시각화용 데이터도 함께 반환한다.
    """
    scores = {}  # key=pivot['time'], value=점수
    visuals = []   # 시각화용 데이터 목록
    N = len(pivots)
    if N < 3:
        return scores, visuals

    for i in range(N - 2):
        P1 = pivots[i]
        P2 = pivots[i+1]

        if P1['type'] == P2['type']:
            continue  # 서로 다른 타입이어야 첫 충격파동 성립

        is_up = (P1['type'] == 'T' and P2['type'] == 'P')
        impulse = abs(P2['price'] - P1['price'])
        if impulse <= 0:
            continue

        # 되돌림 존 50~82% (P2 기준, 충격 반대방향)
        if is_up:
            Z50 = P2['price'] - 0.50 * impulse
            Z82 = P2['price'] - 0.82 * impulse  # 하단 경계
        else:
            Z50 = P2['price'] + 0.50 * impulse
            Z82 = P2['price'] + 0.82 * impulse  # 상단 경계

        # P2 가격 초과(패턴 무효화) 전까지의 스캔 종료 인덱스 찾기
        break_idx = None
        for k in range(i + 2, N):
            pk = pivots[k]
            if is_up and pk['price'] > P2['price'] + EPS:
                break_idx = k
                break
            if not is_up and pk['price'] < P2['price'] - EPS:
                break_idx = k
                break
        
        end_idx = break_idx if break_idx is not None else N - 1

        # --- 점수 계산 시작 ---

        # A) 존 히트 피봇에 +1점
        for j in range(i + 2, end_idx + 1):
            pj = pivots[j]
            # 상승은 저점(T) 히트, 하락은 고점(P) 히트여야 함
            if (is_up and pj['type'] != 'T') or (not is_up and pj['type'] != 'P'):
                continue

            in_zone = (min(Z50, Z82) - EPS <= pj['price'] <= max(Z50, Z82) + EPS)
            if in_zone:
                scores[pj['time']] = scores.get(pj['time'], 0) + 1
                # 시각화 데이터는 필요 시 여기서 추가 (visuals.append(...))

        # B) P2 초과 이전, 82% 경계 이탈 피봇에 -1점
        for j in range(i + 2, end_idx + 1):
            pj = pivots[j]
            if is_up:
                # 하락 되돌림이 82% 하단 경계를 더 깊게 침투
                if pj['type'] == 'T' and pj['price'] < Z82 - EPS:
                    scores[pj['time']] = scores.get(pj['time'], 0) - 1
            else:
                # 상승 되돌림이 82% 상단 경계를 초과
                if pj['type'] == 'P' and pj['price'] > Z82 + EPS:
                    scores[pj['time']] = scores.get(pj['time'], 0) - 1

        # C) P2 초과 이후, 150% 확장 달성 시 +1점
        if break_idx is not None:
            if is_up:
                target_ext = P2['price'] + 1.50 * impulse
                # 브레이크 이후 구간에서 목표 달성한 첫 피봇 찾기
                ext_pivot = next((p for p in pivots[break_idx:] if p['price'] >= target_ext - EPS), None)
            else:
                target_ext = P2['price'] - 1.50 * impulse
                ext_pivot = next((p for p in pivots[break_idx:] if p['price'] <= target_ext + EPS), None)
            
            if ext_pivot is not None:
                scores[ext_pivot['time']] = scores.get(ext_pivot['time'], 0) + 1

    return scores, visuals

# ==============================================================================
# ## 섹션 3: 메인 실행 로직
# ==============================================================================
if __name__ == '__main__':
    # --- 설정 변수 ---
    START_DATE_STR = '2019-12-01'
    END_DATE_STR = '2019-12-31'
    TIMEFRAME = "15m"
    SYMBOL = "BTCUSDT"
    
    # 타임프레임별 권장 설정
    chunk_map = {'1m': '15D', '5m': '30D', '15m': '45D', '1h': '90D', '4h': '180D', '1d': '365D', '1W': '1500D', '1M': '5000D'}
    lookaround_map = {'1m': 12, '5m': 8, '15m': 5, '1h': 5, '4h': 4, '1d': 3, '1W': 2, '1M': 2}
    
    CHUNK_FREQUENCY = chunk_map.get(TIMEFRAME.upper(), '365D')
    LOOKAROUND = lookaround_map.get(TIMEFRAME, 3)
    TOLERANCE = 0.001  # 부동소수점 오차 허용치 (EPS)
    OUTPUT_FILENAME = f"{TIMEFRAME}_impulse_scored_pivots_{START_DATE_STR}_to_{END_DATE_STR}.parquet"

    print("="*60)
    print("--- 배치 분석 시작 (v7 - 충격파동 시퀀스 스코어링) ---")
    print(f"심볼: {SYMBOL}, 타임프레임: {TIMEFRAME}")
    print(f"분석 기간: {START_DATE_STR} ~ {END_DATE_STR}")
    print(f"출력 파일: {OUTPUT_FILENAME}")
    print("="*60)
    
    date_chunks = pd.date_range(start=START_DATE_STR, end=END_DATE_STR, freq=CHUNK_FREQUENCY)
    if pd.to_datetime(END_DATE_STR) not in date_chunks:
        date_chunks = date_chunks.append(pd.Index([pd.to_datetime(END_DATE_STR)]))

    writer = None

    for i in range(len(date_chunks) - 1):
        chunk_start_date, chunk_end_date = date_chunks[i], date_chunks[i+1]
        chunk_start_str = chunk_start_date.strftime('%Y-%m-%d')
        start_ts, end_ts = int(chunk_start_date.timestamp() * 1000), int(chunk_end_date.timestamp() * 1000)

        print(f"\n--- [{i+1}/{len(date_chunks)-1}] 조각 처리 중: {chunk_start_str} ~ {chunk_end_date.strftime('%Y-%m-%d')} ---")
        
        df = fetch_klines(SYMBOL, TIMEFRAME, start_ts, end_ts)
        if df.empty or len(df) <= LOOKAROUND * 2:
            print("데이터 부족으로 건너뜁니다.")
            continue

        pivots = find_pivots_optimized(df, LOOKAROUND)
        if not pivots or len(pivots) < 3:
            print("피봇을 찾지 못했거나 수가 부족하여 건너뜁니다.")
            continue
        
        print(f"피봇 {len(pivots)}개를 찾았습니다. 스코어링을 시작합니다...")
        
        # 새로운 스코어링 함수 호출
        scores, visuals = score_visual_first_impulse(pivots, TOLERANCE)
        
        if not scores:
            print("점수가 매겨진 피봇이 없어 다음 조각으로 넘어갑니다.")
            continue
            
        print(f"총 {len(scores)}개의 피봇에 대한 점수를 계산했습니다.")

        # 결과를 DataFrame으로 변환
        pivots_df = pd.DataFrame(pivots)
        pivots_df['score'] = pivots_df['time'].map(scores).fillna(0).astype(int)
        
        # 점수가 0이 아닌 피봇만 저장 (필요에 따라 주석 처리/해제)
        # pivots_df = pivots_df[pivots_df['score'] != 0]

        if not pivots_df.empty:
            table = pa.Table.from_pandas(pivots_df, preserve_index=False)
            
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_FILENAME, table.schema)
            
            writer.write_table(table)
            print(f"결과가 '{OUTPUT_FILENAME}'에 추가되었습니다.")
        else:
            print("해당 조각에서 유효한 점수를 가진 피봇을 찾지 못했습니다.")

    if writer:
        writer.close()
        print(f"\n--- 모든 조각 처리 완료. 최종 파일 '{OUTPUT_FILENAME}'이 생성되었습니다. ---")
    else:
        print("\n분석할 데이터가 없어 작업을 종료합니다.")