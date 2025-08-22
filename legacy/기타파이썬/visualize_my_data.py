import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import json

# 섹션 1: 상수 및 전역 설정
EPSILON = 1e-6

# ==============================================================================
# ## 섹션 2: 헬퍼 함수
# ==============================================================================
def detect_pivots(df, order=5):
    """OHLC 데이터프레임에서 피봇 포인트를 탐지하는 함수"""
    pivots = []
    if len(df) < order * 2 + 1:
        return []

    price_range = df['high'].max() - df['low'].min()
    if price_range < EPSILON:
        return []

    prominence = price_range * 0.01

    high_idx, _ = find_peaks(df['high'], distance=order, prominence=prominence)
    low_idx, _ = find_peaks(-df['low'], distance=order, prominence=prominence)

    for idx in high_idx:
        pivots.append({'time': df.index[idx], 'price': df['high'].iloc[idx], 'type': 'P'})
    for idx in low_idx:
        pivots.append({'time': df.index[idx], 'price': df['low'].iloc[idx], 'type': 'T'})

    pivots.sort(key=lambda x: x['time'])
    return pivots

# ✅ 새로운 함수: 1분봉 되돌림 내부의 되돌림을 분석
def analyze_nested_retracement(df_1m_slice, direction):
    """
    1분봉 슬라이스 내에서 되돌림이 40-82% 존에 닿았는지 체크하는 함수.
    이 함수는 15분봉 임펄스에 대한 1분봉 되돌림 구간을 받아서,
    그 안의 작은 파동(임펄스)의 되돌림을 분석한다.
    """
    internal_pivots = detect_pivots(df_1m_slice, order=3)
    if len(internal_pivots) < 3:
        return False

    # 1분봉 되돌림 안의 첫 번째 임펄스를 찾는다
    # Long: P-T-P or P-T
    # Short: T-P-T or T-P
    for i in range(len(internal_pivots) - 2):
        p1_internal = internal_pivots[i]
        p2_internal = internal_pivots[i+1]
        p3_internal = internal_pivots[i+2]

        if p1_internal['type'] == p2_internal['type']:
            continue

        # 1분봉 임펄스 크기
        internal_impulse_size = abs(p2_internal['price'] - p1_internal['price'])
        if internal_impulse_size < EPSILON:
            continue

        # 1분봉 임펄스에 대한 되돌림 비율 계산
        if p1_internal['type'] == 'T' and p2_internal['type'] == 'P' and p3_internal['type'] == 'T': # T-P-T 패턴 (Long)
            retrace_ratio = (p2_internal['price'] - p3_internal['price']) / internal_impulse_size
            if 0.40 <= retrace_ratio <= 0.82:
                print("1분봉 되돌림 안에서 40-82% 되돌림 패턴 발견.")
                return True
        elif p1_internal['type'] == 'P' and p2_internal['type'] == 'T' and p3_internal['type'] == 'P': # P-T-P 패턴 (Short)
            retrace_ratio = (p3_internal['price'] - p2_internal['price']) / internal_impulse_size
            if 0.40 <= retrace_ratio <= 0.82:
                print("1분봉 되돌림 안에서 40-82% 되돌림 패턴 발견.")
                return True

    return False

def analyze_and_score_patterns(pivots_15m: list, df_1m: pd.DataFrame) -> list:
    """
    15분봉 임펄스를 1분봉으로 세분화하여 분석하고 점수를 매기는 함수 (새로운 로직).
    """
    scored_impulses = []
    print("15분봉 임펄스 -> 1분봉 세부 분석 및 점수화 시작...")

    for i in range(len(pivots_15m) - 1):
        p1 = pivots_15m[i]
        p2 = pivots_15m[i+1]

        if p1['type'] == p2['type']:
            continue

        direction = "Long" if p1['type'] == 'T' and p2['type'] == 'P' else "Short"
        impulse_size = abs(p2['price'] - p1['price'])
        
        # 15분봉 임펄스에 대한 50-82% 되돌림 존
        if direction == "Long":
            retrace_zone_high_15m = p2['price'] - 0.50 * impulse_size
            retrace_zone_low_15m = p2['price'] - 0.82 * impulse_size
        else: # Short
            retrace_zone_low_15m = p2['price'] + 0.50 * impulse_size
            retrace_zone_high_15m = p2['price'] + 0.82 * impulse_size

        retrace_start_time = p2['time']
        retrace_end_time = pivots_15m[i+2]['time'] if i + 2 < len(pivots_15m) else df_1m.index[-1]

        df_1m_slice = df_1m[(df_1m.index > retrace_start_time) & (df_1m.index <= retrace_end_time)]

        if df_1m_slice.empty:
            continue

        impulse_info = {
            'p1_time': p1['time'],
            'p2_time': p2['time'],
            'direction': direction,
            'status': None,
            'score': np.nan
        }

        # 1. 15분봉 임펄스의 82% 되돌림 존을 이탈했는지 체크 (FAIL_BREAK)
        fail_break = False
        if direction == "Long":
            if df_1m_slice['low'].min() < retrace_zone_low_15m:
                fail_break = True
        else: # Short
            if df_1m_slice['high'].max() > retrace_zone_high_15m:
                fail_break = True
        
        if fail_break:
            impulse_info['status'] = "FAIL_BREAK"
            impulse_info['score'] = -1
        
        # 2. 15분봉 되돌림이 50-82% 존에 닿았고 & 그 되돌림 안에서 1분봉 되돌림이 40-82% 존에 닿았는지 체크 (SUCCESS)
        else:
            has_retrace_into_zone_15m = False
            if direction == "Long":
                if df_1m_slice['low'].min() <= retrace_zone_high_15m: # 50% 존 안으로 들어왔는가?
                    has_retrace_into_zone_15m = True
            else: # Short
                if df_1m_slice['high'].max() >= retrace_zone_low_15m: # 50% 존 안으로 들어왔는가?
                    has_retrace_into_zone_15m = True

            if has_retrace_into_zone_15m:
                # 1분봉 되돌림 안의 프랙탈 되돌림을 체크
                if analyze_nested_retracement(df_1m_slice, direction):
                    # 모든 조건 만족 시, 새로운 고점/저점 갱신 여부로 최종 SUCCESS 판정
                    success = False
                    if direction == "Long":
                        if df_1m_slice['high'].max() > p2['price']:
                            success = True
                    else:
                        if df_1m_slice['low'].min() < p2['price']:
                            success = True

                    if success:
                        impulse_info['status'] = "SUCCESS"
                        impulse_info['score'] = 1
                    else:
                        impulse_info['status'] = "FAIL_EXTREME"
                        impulse_info['score'] = 0.5
                else: # 1분봉 되돌림 안의 되돌림이 조건을 충족하지 못한 경우
                    impulse_info['status'] = "FAIL_EXTREME"
                    impulse_info['score'] = 0.5
            else: # 15분봉 되돌림이 50% 존에 도달하지 못한 경우
                impulse_info['status'] = "FAIL_EXTREME"
                impulse_info['score'] = 0.5

        scored_impulses.append(impulse_info)

    print(f"분석 완료. 총 {len(scored_impulses)}개의 15분봉 임펄스에 대한 점수화 완료.")
    return scored_impulses

# ==============================================================================
# ## 섹션 3: 메인 실행 로직
# ==============================================================================
if __name__ == '__main__':
    try:
        print("15분봉, 1분봉 Parquet 파일 로딩...")
        df_15m_raw = pd.read_parquet("15m_impulse_scored_pivots_2019-12-01_to_2024-12-31.parquet")
        df_1m_raw = pd.read_parquet("1m_impulse_scored_pivots_2019-12-01_to_2024-12-31.parquet")

        df_15m_raw['time'] = pd.to_datetime(df_15m_raw['time'], unit='ms')
        df_1m_raw['time'] = pd.to_datetime(df_1m_raw['time'], unit='ms')

        df_15m_raw.set_index('time', inplace=True)
        df_1m_raw.set_index('time', inplace=True)

        print("OHLC 데이터 생성 중...")
        df_15m = df_15m_raw['price'].resample('15min').ohlc().dropna()
        df_1m = df_1m_raw['price'].resample('1min').ohlc().dropna()
        print("파일 로딩 및 OHLC 데이터 변환 완료.")

        if len(df_15m) < 20 or len(df_1m) < 60:
            print("[오류] 데이터 변환 후 데이터가 너무 적습니다. 원본 파일을 확인하세요.")
            exit()

    except FileNotFoundError:
        print("[오류] Parquet 파일을 찾을 수 없습니다. 파일 경로를 확인하세요.")
        exit()
    except Exception as e:
        print(f"데이터 처리 중 심각한 오류 발생: {e}")
        exit()

    print("\n15분봉 피봇 탐지 중...")
    pivots_15m = detect_pivots(df_15m)
    print(f"15분봉 피봇 {len(pivots_15m)}개 탐지 완료.")

    if len(pivots_15m) < 3:
        print("\n탐지된 15분봉 피봇이 너무 적어 분석을 진행할 수 없습니다.")
        exit()

    scored_patterns = analyze_and_score_patterns(pivots_15m, df_1m)

    if scored_patterns:
        print("\n--- 패턴 분석 및 점수화 결과 (KST) ---")
        print(f"{'번호':<5} | {'임펄스 시작 (P1)':<28} | {'방향':<7} | {'상태':<15} | {'점수'}")
        print("-" * 80)
        
        results_for_json = []
        for i, pattern in enumerate(scored_patterns):
            # 화면 출력용 시간 변환
            p1_time_kst = pattern['p1_time'].tz_localize('UTC').tz_convert('Asia/Seoul')
            p1_time_str = p1_time_kst.strftime('%Y-%m-%d %H:%M %Z')
            print(f"{i+1:<5} | {p1_time_str:<28} | {pattern['direction']:<7} | {pattern['status']:<15} | {pattern['score']}")
            
            # JSON 저장용 데이터 가공 (ISO 형식으로 시간 저장)
            p2_time_kst = pattern['p2_time'].tz_localize('UTC').tz_convert('Asia/Seoul')
            json_pattern = {
                'p1_time_kst': p1_time_kst.isoformat(),
                'p2_time_kst': p2_time_kst.isoformat(),
                'direction': pattern['direction'],
                'status': pattern['status'],
                'score': pattern['score']
            }
            results_for_json.append(json_pattern)
            
        # 결과를 JSON 파일로 저장
        output_filename = 'analysis_results.json'
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(results_for_json, f, ensure_ascii=False, indent=4)
            print(f"\n분석 결과를 '{output_filename}' 파일에 JSON 형식으로 저장했습니다.")
        except Exception as e:
            print(f"\n[오류] 결과를 JSON 파일로 저장하는 중 오류 발생: {e}")