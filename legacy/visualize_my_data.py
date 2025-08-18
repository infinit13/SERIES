import pandas as pd
import mplfinance as mpf
import os
import requests
from datetime import datetime
import numpy as np
from collections import defaultdict
import re
import time

# ==============================================================================
# ## 섹션 1: 설정
# ==============================================================================
PARQUET_FILENAME = '15m_impulse_scored_pivots_2019-12-01_to_2019-12-31.parquet'
CHART_OUTPUT_DIR = 'charts_on_demand'

# ==============================================================================
# ## 섹션 2: '캠페인' 스코어링 함수 (시각화 간소화 버전)
# ==============================================================================
# 로직:
# 1. P1->P2 임펄스 후, 되돌림 존에 최초 진입하는 피봇을 내부적으로 찾는다. (녹색 점선 제거)
# 2. 그 이후, 존의 하단을 깨면 실패로 간주하고 추적을 종료한다. (빨간색 점선 제거)
# 3. 존을 깨지 않은 상태에서 P2를 돌파하면 성공으로 간주하고, 그 돌파 피봇에 +1점을 부여하고 청록색 점선만 표시한다.
def score_campaign_patterns(pivots: list, EPS: float) -> tuple[dict, dict, list]:
    visuals = []
    long_scores = {}
    short_scores = {}
    N = len(pivots)
    if N < 3:
        return {}, {}, []

    for i in range(N - 2):
        P1, P2 = pivots[i], pivots[i+1]
        
        if P1['type'] == P2['type']: continue
        is_up = (P1['type'] == 'T' and P2['type'] == 'P')
        impulse = abs(P2['price'] - P1['price'])
        if impulse <= EPS: continue
        
        visuals.append({'type': 'line', 'x0': P1['time'], 'y0': P1['price'], 'x1': P2['time'], 'y1': P2['price'], 'color': 'blue', 'style': 'solid'})

        if is_up:
            Z50, Z82 = P2['price'] - 0.50 * impulse, P2['price'] - 0.82 * impulse
            DIR = 'L'
        else:
            Z50, Z82 = P2['price'] + 0.50 * impulse, P2['price'] + 0.82 * impulse
            DIR = 'S'
            
        scan_end_idx = next((k for k, pk in enumerate(pivots[i+2:], i+2) if (is_up and pk['price'] > P2['price'] + EPS) or (not is_up and pk['price'] < P2['price'] - EPS)), N)
        scan_end_time = pivots[scan_end_idx - 1]['time'] if scan_end_idx < N else pivots[-1]['time']
        visuals.append({'type': 'band', 'x0': P2['time'], 'x1': scan_end_time, 'y0': min(Z50, Z82), 'y1': max(Z50, Z82), 'color': 'gray', 'alpha': 0.15})

        # --- 1단계: 최초 되돌림 피봇 찾기 ---
        pj_retracement, j_retracement_idx = None, -1
        for j in range(i + 2, N):
            pj = pivots[j]
            if ((is_up and pj['type'] == 'T') or (not is_up and pj['type'] == 'P')) and \
               (min(Z50, Z82) - EPS <= pj['price'] <= max(Z50, Z82) + EPS):
                pj_retracement, j_retracement_idx = pj, j
                # visuals.append({'type': 'line', 'x0': P2['time'], 'y0': P2['price'], 'x1': pj['time'], 'y1': pj['price'], 'color': 'green', 'style': 'dashed'}) # 녹색 점선 제거
                break
        
        if not pj_retracement:
            continue

        # --- 2단계: 되돌림 이후 '실패(존 하향 이탈)' 또는 '성공(P2 돌파)' 추적 ---
        for k in range(j_retracement_idx + 1, N):
            pk = pivots[k]
            
            # 실패 조건: 존의 하단(Z82)을 깨는 새로운 저점/고점이 출현
            is_failure = (is_up and pk['type'] == 'T' and pk['price'] < Z82 - EPS) or \
                         (not is_up and pk['type'] == 'P' and pk['price'] > Z82 + EPS)
            if is_failure:
                # visuals.append({'type': 'hline', 'y': Z82, 'x0': pj_retracement['time'], 'x1': pk['time'], 'color': 'red', 'style': 'dotted'}) # 빨간색 점선 제거
                break

            # 성공 조건: P2 가격을 돌파
            is_success = (is_up and pk['price'] > P2['price'] + EPS) or \
                         (not is_up and pk['price'] < P2['price'] - EPS)
            if is_success:
                if DIR == 'L': long_scores[pk['time']] = 1
                else: short_scores[pk['time']] = 1
                visuals.append({'type': 'line', 'x0': pj_retracement['time'], 'y0': pj_retracement['price'], 'x1': pk['time'], 'y1': pk['price'], 'color': 'cyan', 'style': 'dotted'})
                break
                
    return long_scores, short_scores, visuals

# ==============================================================================
# ## 섹션 3: 헬퍼 및 시각화 함수
# ==============================================================================
def fetch_klines(symbol: str, timeframe: str, start_ts_ms: int, end_ts_ms: int) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_ts = start_ts_ms
    print(f"바이낸스 서버에서 {symbol}({timeframe}) 데이터를 요청합니다...")
    
    while current_ts < end_ts_ms:
        params = {"symbol": symbol.upper(), "interval": timeframe, "startTime": current_ts, "endTime": end_ts_ms, "limit": 1000}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data: break 
            all_data.extend(data)
            last_timestamp = data[-1][0]
            current_ts = last_timestamp + 1
            print(f"... {datetime.fromtimestamp(last_timestamp/1000)} 까지 데이터 수신 완료.")
            time.sleep(0.1)
        except requests.exceptions.RequestException as e:
            print(f"데이터 가져오기 오류: {e}")
            return pd.DataFrame()
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    df = df[~df.index.duplicated(keep='first')]
    start_dt, end_dt = pd.to_datetime(start_ts_ms, unit='ms'), pd.to_datetime(end_ts_ms, unit='ms')
    return df[(df.index >= start_dt) & (df.index <= end_dt)]

def visualize_all_patterns(df: pd.DataFrame, pivots_df: pd.DataFrame, visuals: list, chart_title: str, output_dir: str, start_date: datetime, end_date: datetime, timeframe: str):
    if df.empty: 
        print("경고: 시각화할 캔들 데이터가 비어있습니다.")
        return
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    pivots_df = pivots_df.set_index('time')
    
    high_pivots_aligned, low_pivots_aligned = pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
    valid_high_pivots = pivots_df[(pivots_df['type'] == 'P') & pivots_df.index.isin(df.index)]
    valid_low_pivots = pivots_df[(pivots_df['type'] == 'T') & pivots_df.index.isin(df.index)]
    high_pivots_aligned.loc[valid_high_pivots.index] = valid_high_pivots['price']
    low_pivots_aligned.loc[valid_low_pivots.index] = valid_low_pivots['price']
    
    apds = [ 
        mpf.make_addplot(high_pivots_aligned, type='scatter', marker='v', color='tomato', markersize=60),
        mpf.make_addplot(low_pivots_aligned, type='scatter', marker='^', color='limegreen', markersize=60),
    ]
    
    y_min, y_max = df['low'].min(), df['high'].max()
    padding = (y_max - y_min) * 0.05 
    ylim = (y_min - padding, y_max + padding * 2)
    
    fig, axes = mpf.plot(df, type='candle', style='yahoo', title=f"{chart_title} ({timeframe})", ylabel='Price ($)', 
                         addplot=apds, returnfig=True, figsize=(24, 12), xrotation=20)
    ax = axes[0]
    
    for v in visuals:
        x0_dt, x1_dt = pd.to_datetime(v['x0'], unit='ms'), pd.to_datetime(v['x1'], unit='ms')
        if x0_dt <= df.index.max() and x1_dt >= df.index.min():
            if v['type'] == 'line': ax.plot([x0_dt, x1_dt], [v['y0'], v['y1']], color=v['color'], linestyle=v['style'], alpha=0.7, lw=1.5)
            elif v['type'] == 'band': ax.fill_between([x0_dt, x1_dt], v['y0'], v['y1'], color=v['color'], alpha=v['alpha'], interpolate=True)
            elif v['type'] == 'hline': ax.plot([x0_dt, x1_dt], [v['y'], v['y']], color=v['color'], linestyle=v['style'], alpha=0.9, lw=1.5)
            
    for idx, row in pivots_df.iterrows():
        if idx in df.index:
            l_score, s_score = int(row['long_score']), int(row['short_score'])
            text = f"L:{l_score} S:{s_score}"
            
            color = 'grey'
            if l_score > 0:
                color = 'green'
            elif s_score > 0:
                color = 'red'
            
            y_pos = row['price'] * 1.005 if row['type'] == 'P' else row['price'] * 0.995
            va_align = 'bottom' if row['type'] == 'P' else 'top'
            
            ax.text(idx, y_pos, f' {text}', color=color, fontsize=9, fontweight='bold', ha='left', va=va_align)
            
    ax.set_ylim(ylim)
    time_unit = ''.join(re.findall(r'[a-zA-Z]', timeframe))
    padding_delta = pd.Timedelta(hours=6) if time_unit == 'm' else pd.Timedelta(hours=24) if time_unit == 'h' else pd.Timedelta(days=2)
    ax.set_xlim(start_date - padding_delta, end_date + padding_delta)
    
    filename = os.path.join(output_dir, f'chart_{chart_title.replace(" ", "_")}.png')
    fig.savefig(filename)
    print(f"성공! '{filename}'에 모든 패턴을 포함한 차트를 저장했습니다.")

# ==============================================================================
# ## 섹션 4: 메인 실행 로직
# ==============================================================================
if __name__ == '__main__':
    try:
        tf_match = re.search(r'(\d+[a-zA-Z])', PARQUET_FILENAME)
        if not tf_match: raise ValueError("파일명에서 타임프레임(예: 15m, 1h, 1d)을 찾을 수 없습니다.")
        TIMEFRAME = tf_match.group(1)
        print(f"'{PARQUET_FILENAME}' 파일 로딩 중... 감지된 타임프레임: {TIMEFRAME}")
        pivots_df_full = pd.read_parquet(PARQUET_FILENAME)
        pivots_df_full['time'] = pd.to_datetime(pivots_df_full['time'], unit='ms')
        print("파일 로딩 및 시간 변환 완료.")
    except (FileNotFoundError, ValueError) as e:
        print(f"오류: {e}")
        exit()

    pivots_df_for_analysis = pivots_df_full.copy()
    pivots_df_for_analysis['time'] = (pivots_df_for_analysis['time'].astype(np.int64) // 10**6)
    all_pivots_list = pivots_df_for_analysis.to_dict('records')
    EPSILON = 1e-9
    
    print("전체 피봇에 대한 '캠페인' 스코어링 및 시각화 데이터 생성 중...")
    all_long_scores, all_short_scores, all_visuals = score_campaign_patterns(all_pivots_list, EPSILON)
    
    time_as_int_ms = (pivots_df_full['time'].astype(np.int64) // 10**6)
    pivots_df_full['long_score'] = time_as_int_ms.map(all_long_scores).fillna(0).astype(int)
    pivots_df_full['short_score'] = time_as_int_ms.map(all_short_scores).fillna(0).astype(int)
    
    print("\n" + "="*60)
    print("=== 인터랙티브 차트 생성 모드 (날짜/시간 입력) ===")
    print(f"현재 분석 대상 타임프레임: {TIMEFRAME}")
    print("날짜/시간을 입력하여 해당 기간의 분석 차트를 생성합니다. (예: 2019-12-15 09:00)")
    print(f"차트는 '{CHART_OUTPUT_DIR}' 폴더에 저장됩니다.")
    print("종료하려면 'q' 또는 'exit'를 입력하세요.")
    print("="*60)
    
    while True:
        try:
            start_input = input(f"조회할 시작 날짜 (YYYY-MM-DD HH:MM): ")
            if start_input.lower() in ['q', 'exit']: break
            end_input = input(f"조회할 종료 날짜 (YYYY-MM-DD HH:MM): ")
            if end_input.lower() in ['q', 'exit']: break
            start_date, end_date = pd.to_datetime(start_input), pd.to_datetime(end_input)
            start_ts_int, end_ts_int = int(start_date.timestamp() * 1000), int(end_date.timestamp() * 1000)
            pivots_slice = pivots_df_full[(pivots_df_full['time'] >= start_date) & (pivots_df_full['time'] <= end_date)]
            df_slice = fetch_klines("BTCUSDT", TIMEFRAME, start_ts_int, end_ts_int)
            if df_slice.empty:
                print("해당 기간에 캔들 데이터가 없습니다.")
                continue
            visuals_slice = [v for v in all_visuals if pd.to_datetime(v['x0'], unit='ms') <= end_date and pd.to_datetime(v['x1'], unit='ms') >= start_date]
            chart_title = f"Analysis_{start_date.strftime('%Y%m%d%H%M')}_to_{end_date.strftime('%Y%m%d%H%M')}"
            visualize_all_patterns(df_slice, pivots_slice, visuals_slice, chart_title, CHART_OUTPUT_DIR, start_date, end_date, TIMEFRAME)
        except ValueError:
            print("입력 오류: 'YYYY-MM-DD HH:MM' 형식에 맞게 입력해야 합니다.")
        except (KeyboardInterrupt, EOFError):
            break
            
    print("\n분석 도구를 종료합니다.")