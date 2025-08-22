# view_parquet.py

import pandas as pd
import argparse
import os

def view_parquet(filepath, head=None):
    """
    Parquet 파일을 읽어 터미널에 내용과 요약 정보를 출력합니다.
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
    total_rows = len(df)
    columns = df.columns.tolist()
    
    print("\n" + "="*50)
    print(" 파일 요약 (File Summary)")
    print("="*50)
    print(f"  - 총 행(Rows) 수: {total_rows}")
    print(f"  - 열(Columns) 목록: {columns}")
    print("="*50 + "\n")

    # --- 3. 데이터 출력 ---
    if df.empty:
        print("결과: 파일에 데이터가 없습니다.")
        return
        
    # 터미널에 모든 컬럼을 보기 좋게 출력하기 위한 설정
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.width', 120) # 터미널 너비에 맞게 조절
    pd.set_option('display.max_columns', None) # 모든 열을 표시

    if head:
        print(f"--- 상위 {head}개 행 표시 ---")
        print(df.head(head))
    else:
        print("--- 전체 데이터 표시 ---")
        print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parquet 파일 터미널 뷰어")
    
    parser.add_argument("filepath", type=str, help="내용을 확인할 Parquet 파일의 경로")
    
    # 옵션 인수 추가
    parser.add_argument("--head", "-n", type=int, help="상위 N개의 행만 표시합니다. 예: -n 20")
    
    args = parser.parse_args()
    
    view_parquet(args.filepath, args.head)