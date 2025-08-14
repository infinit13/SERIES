import pandas as pd
import argparse
import os

def explore_pivot_data(filepath, series_id=None, min_score=None, head=None):
    """
    분석이 완료된 피봇 로그 Parquet 파일을 읽고,
    기본 정보 출력 및 필터링을 수행하는 탐색기 스크립트.
    """
    # --- 1. 파일 존재 여부 확인 ---
    if not os.path.exists(filepath):
        print(f"오류: 파일을 찾을 수 없습니다 - '{filepath}'")
        return

    print(f"'{filepath}' 파일 로딩 중...")
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"파일을 읽는 중 오류 발생: {e}")
        return

    # --- 2. 기본 정보 출력 ---
    total_pivots = len(df)
    unique_series = df['series_id'].nunique()
    
    print("\n" + "="*50)
    print(" 데이터 요약 (Summary)")
    print("="*50)
    print(f"  - 총 피봇 로그 수: {total_pivots}")
    print(f"  - 총 시리즈 개수: {unique_series}")
    if not df.empty:
        min_date = pd.to_datetime(df['timestamp'].min(), unit='ms').strftime('%Y-%m-%d')
        max_date = pd.to_datetime(df['timestamp'].max(), unit='ms').strftime('%Y-%m-%d')
        print(f"  - 데이터 기간: {min_date} ~ {max_date}")
    print("="*50 + "\n")

    # --- 3. 데이터 필터링 ---
    # 여러 필터를 순차적으로 적용
    filtered_df = df
    
    if series_id:
        filtered_df = filtered_df[filtered_df['series_id'] == series_id]
        print(f"-> '{series_id}' 시리즈로 필터링 중...")

    if min_score is not None:
        filtered_df = filtered_df[filtered_df['cumulative_score'] >= min_score]
        print(f"-> 누적 점수(cumulative_score)가 {min_score} 이상인 피봇으로 필터링 중...")
        
    # --- 4. 결과 출력 ---
    if filtered_df.empty:
        print("결과: 필터 조건에 맞는 데이터가 없습니다.")
        return
        
    print(f"결과: {len(filtered_df)}개의 피봇 로그 발견\n")

    # 터미널에 모든 컬럼을 보기 좋게 출력하기 위한 설정
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 15)

    if head:
        print(filtered_df.head(head))
    else:
        print(filtered_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전술적 피봇 로그(.parquet) 탐색기")
    
    parser.add_argument("filepath", type=str, help="분석할 Parquet 파일의 경로")
    
    # 옵션 인수 추가
    parser.add_argument("--series_id", type=str, help="특정 series_id만 필터링합니다. 예: '20220101_1h_42'")
    parser.add_argument("--min_score", type=float, help="표시할 최소 누적 점수를 지정합니다. 예: 10.0")
    parser.add_argument("--head", type=int, help="상위 N개의 결과만 표시합니다.")
    
    args = parser.parse_args()
    
    explore_pivot_data(args.filepath, args.series_id, args.min_score, args.head)