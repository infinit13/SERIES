import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime
import time
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==============================================================================
# ## 섹션 1: 설정
# ==============================================================================
TF_GRANDPARENT = '4h'
TF_PARENT = '1h'
TF_CHILD = '15m' 

# [수정] 결과 미리보기 개수 설정
PREVIEW_COUNT = 20

# 차트를 저장할 디렉토리
CHART_OUTPUT_DIR = 'chart_previews'

def get_parquet_path(timeframe):
    if timeframe == '15m':
        # 실제 사용하는 15m 피봇 파일명으로 변경해야 함
        return '15m_impulse_scored_pivots_2019-12-01_to_2024-12-31.parquet'
    if timeframe == '5m': return 'analysis_results_5years_robust.parquet'
    return f'{timeframe}_analysis_results_5years_robust.parquet'

OUTPUT_SUFFIX = '_with_impulse_counts'
EPSILON = 1e-9

# ==============================================================================
# ## 섹션 2: 헬퍼 함수 (데이터 처리 및 API 호출)
# ==============================================================================
def find_impulse_sections(child_df: pd.DataFrame) -> list:
    print(f"'{TF_CHILD}' 타임프레임에서 P1->P2 임펄스 구간을 추출합니다...")
    child_df['time'] = pd.to_datetime(child_df['time'], unit='ms')
    pivots = child_df.to_dict('records')
    impulses = []
    for i in range(len(pivots) - 1):
        P1, P2 = pivots[i], pivots[i+1]
        if P1['type'] != P2['type'] and abs(P2['price'] - P1['price']) > EPSILON:
            start_ts = int(pd.Timestamp(P1['time']).value / 1_000_000)
            end_ts = int(pd.Timestamp(P2['time']).value / 1_000_000)
            impulses.append({'start_ts': start_ts, 'end_ts': end_ts})
    print(f"총 {len(impulses)}개의 임펄스 구간을 찾았습니다.")
    return impulses

def aggregate_impulses_to_parent(parent_df: pd.DataFrame, impulses: list, timeframe_name: str) -> pd.DataFrame:
    print(f"'{timeframe_name}' 타임프레임에 임펄스 발생 횟수를 집계합니다...")
    impulse_counts = []
    for _, row in parent_df.iterrows():
        count = sum(1 for imp in impulses if row['start_ts'] <= imp['start_ts'] and imp['end_ts'] <= row['end_ts'])
        impulse_counts.append(count)
    parent_df['child_impulse_count'] = impulse_counts
    # [추가] 카운트 기준으로 내림차순 정렬
    parent_df = parent_df.sort_values(by='child_impulse_count', ascending=False).reset_index(drop=True)
    print("집계 및 정렬 완료.")
    return parent_df

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
            current_ts = data[-1][0] + 1
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
    return df

# ==============================================================================
# ## 섹션 3: 결과 미리보기 차트 생성 함수
# ==============================================================================
def generate_preview_charts(df_preview: pd.DataFrame, all_impulses: list, parent_tf: str):
    print("\n" + "="*60)
    print(f"상위 {PREVIEW_COUNT}개 결과에 대한 미리보기 차트를 생성합니다...")
    if not os.path.exists(CHART_OUTPUT_DIR):
        os.makedirs(CHART_OUTPUT_DIR)

    parent_tf_freq = parent_tf.upper()

    # [수정] PREVIEW_COUNT 만큼 루프 실행
    for index, row in df_preview.head(PREVIEW_COUNT).iterrows():
        start_ts, end_ts = int(row['start_ts']), int(row['end_ts'])
        impulse_count = row['child_impulse_count']
        
        padding = (end_ts - start_ts) * 0.2
        chart_start_ts, chart_end_ts = int(start_ts - padding), int(end_ts + padding)

        df_kline = fetch_klines("BTCUSDT", parent_tf, chart_start_ts, chart_end_ts)
        if df_kline.empty:
            print(f"경고: #{index} 패턴의 캔들 데이터를 가져올 수 없습니다.")
            continue
            
        start_dt = pd.to_datetime(start_ts, unit='ms')
        end_dt = pd.to_datetime(end_ts, unit='ms')
        
        relevant_impulses = [imp for imp in all_impulses if start_ts <= imp['start_ts'] and imp['end_ts'] <= end_ts]
        impulse_lines = [pd.to_datetime(imp['start_ts'], unit='ms') for imp in relevant_impulses]
        
        chart_title = f"Rank #{index+1} Parent Pattern (Impulse Count: {impulse_count})\n" \
                      f"{start_dt.strftime('%Y-%m-%d %H:%M')} ~ {end_dt.strftime('%H:%M')}"
        
        fig, axlist = mpf.plot(
            df_kline.asfreq(parent_tf_freq),
            type='candle', style='yahoo', title=chart_title,
            ylabel='Price ($)', figsize=(14, 6),
            returnfig=True, show_nontrading=True
        )
        
        ax = axlist[0]
        ax.axvspan(start_dt, end_dt, color='gray', alpha=0.15)
        for impulse_time in impulse_lines:
            ax.axvline(impulse_time, color='red', linestyle='--', linewidth=0.7, alpha=0.8)

        ax.xaxis.set_major_locator(plt.MaxNLocator(10))

        filename = os.path.join(CHART_OUTPUT_DIR, f'preview_rank_{index+1:02d}.png')
        fig.savefig(filename)
        print(f" -> 성공! '{filename}' 파일에 차트를 저장했습니다.")
        plt.close(fig)

# ==============================================================================
# ## 섹션 4: 메인 실행 로직
# ==============================================================================
if __name__ == '__main__':
    try:
        print("="*60)
        print("데이터 파일을 로드합니다...")
        df_gp = pd.read_parquet(get_parquet_path(TF_GRANDPARENT))
        df_p = pd.read_parquet(get_parquet_path(TF_PARENT))
        df_c_pivots = pd.read_parquet(get_parquet_path(TF_CHILD))

        print(f"조부모({TF_GRANDPARENT}): {len(df_gp)}개 패턴")
        print(f"부모({TF_PARENT}): {len(df_p)}개 패턴")
        print(f"자식({TF_CHILD}): {len(df_c_pivots)}개 피봇")
        print("="*60)

        child_impulses = find_impulse_sections(df_c_pivots)
        print("="*60)
        
        df_p_updated = aggregate_impulses_to_parent(df_p.copy(), child_impulses, TF_PARENT)
        df_gp_updated = aggregate_impulses_to_parent(df_gp.copy(), child_impulses, TF_GRANDPARENT)
        print("="*60)

        output_path_p = f"{TF_PARENT}{OUTPUT_SUFFIX}.parquet"
        output_path_gp = f"{TF_GRANDPARENT}{OUTPUT_SUFFIX}.parquet"
        df_p_updated.to_parquet(output_path_p)
        df_gp_updated.to_parquet(output_path_gp)
        
        print("분석 완료! 결과가 아래 파일로 저장되었습니다:")
        print(f" -> {output_path_p}")
        print(f" -> {output_path_gp}")
        print(f"\n결과 미리보기 (상위 {PREVIEW_COUNT}개 부모 데이터):")
        # [수정] PREVIEW_COUNT 만큼 텍스트 출력
        print(df_p_updated[['start_ts', 'end_ts', 'child_impulse_count']].head(PREVIEW_COUNT))

        generate_preview_charts(df_p_updated, child_impulses, TF_PARENT)

    except FileNotFoundError as e:
        print(f"오류: 파일 '{e.filename}'을 찾을 수 없습니다.")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")