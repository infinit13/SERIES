import pandas as pd
import numpy as np

# ==============================================================================
# ## 섹션 1: 설정
# ==============================================================================
# 분석할 parquet 파일들의 경로를 설정한다. UMAP6.py와 동일한 파일을 사용한다.
CHILD_TF_PATH = '15m_analysis_results_5years_robust.parquet'
PARENT_TF_PATH = '1h_analysis_results_5years_robust.parquet'
GRANDPARENT_TF_PATH = '4h_analysis_results_5years_robust.parquet'

def get_parquet_path(timeframe):
    """헬퍼 함수: UMAP6.py의 경로 생성 규칙을 그대로 사용"""
    if timeframe == '5m': return 'analysis_results_5years_robust.parquet'
    return f'{timeframe}_analysis_results_5years_robust.parquet'

# UMAP6.py에서 가져온 경로로 재설정 (만약 파일명이 다르다면 이 부분을 수정)
CHILD_TF_PATH = get_parquet_path('15m')
PARENT_TF_PATH = get_parquet_path('1h')
GRANDPARENT_TF_PATH = get_parquet_path('4h')


# ==============================================================================
# ## 섹션 2: 데이터 로드 및 전처리
# ==============================================================================
print("데이터 로딩 중...")
try:
    df_c = pd.read_parquet(CHILD_TF_PATH)
    df_p = pd.read_parquet(PARENT_TF_PATH)
    df_gp = pd.read_parquet(GRANDPARENT_TF_PATH)
    print("데이터 로딩 완료.")
except FileNotFoundError as e:
    print(f"오류: 파일 찾기 실패. '{e.filename}' 파일이 스크립트와 같은 폴더에 있는지 확인하세요.")
    exit()

# visualize_my_data.py의 점수 체계와 유사한 '최종 점수' 컬럼을 가정한다.
# 'direction'은 1.0 (UP/LONG), -1.0 (DOWN/SHORT)으로 가정한다.
# 'retracement_score'가 캠페인의 최종 확정 점수(+1 또는 -1)라고 가정한다.
# 만약 실제 컬럼명이 다르다면 이 부분을 수정해야 한다.
# 예: 최종 점수 컬럼명이 'final_score'라면 df_c['retracement_score'] -> df_c['final_score']
SCORE_COLUMN = 'retracement_score' 

# 시간순으로 정렬
df_c = df_c.sort_values(by='start_ts').reset_index()
df_p = df_p.sort_values(by='start_ts').reset_index()
df_gp = df_gp.sort_values(by='start_ts').reset_index()

# ==============================================================================
# ## 섹션 3: 'L:1 -> L:1' 연속 패턴 탐색
# ==============================================================================
print("\n'L:1 -> L:1' 연속 패턴 탐색 시작...")
l1_sequences = []

# L:1 (롱 캠페인 성공) 패턴만 필터링
# direction == 1.0 (Long) 이고, score == 1 (성공)
df_l1 = df_c[(df_c['direction'] == 1.0) & (df_c[SCORE_COLUMN] == 1)].copy()

# 연속된 L:1 패턴을 찾기 위해 순회
for i in range(len(df_l1) - 1):
    pattern_1 = df_l1.iloc[i]
    pattern_2 = df_l1.iloc[i+1]
    
    # 두 패턴의 index가 연속적인지 확인 (원본 데이터프레임 기준)
    if pattern_2['index'] == pattern_1['index'] + 1:
        # 연속 패턴 발견!
        sequence_info = {
            'seq_start_ts': pattern_1['start_ts'],
            'seq_end_ts': pattern_2['end_ts'],
            'child_1_id': pattern_1['index'],
            'child_2_id': pattern_2['index'],
            'parent_id': None,
            'grandparent_id': None
        }
        l1_sequences.append(sequence_info)

print(f"총 {len(l1_sequences)}개의 'L:1 -> L:1' 연속 패턴을 발견했습니다.")

# ==============================================================================
# ## 섹션 4: 상위 노드 컨텍스트 매핑
# ==============================================================================
print("\n발견된 패턴에 대한 상위 노드 매핑 시작...")
if not l1_sequences:
    print("매핑할 패턴이 없습니다.")
else:
    for seq in l1_sequences:
        # 시퀀스의 종료 시점을 기준으로 어떤 부모/조부모에 속하는지 찾는다.
        target_ts = seq['seq_end_ts']

        # 1. 부모 노드 찾기
        parent_mask = (df_p['start_ts'] <= target_ts) & (df_p['end_ts'] >= target_ts)
        matching_parents = df_p[parent_mask]
        
        if not matching_parents.empty:
            # 여러 부모가 겹칠 경우 가장 마지막에 시작한 부모를 선택 (가장 작은 범위)
            parent_row = matching_parents.sort_values(by='start_ts', ascending=False).iloc[0]
            parent_id = parent_row['index']
            seq['parent_id'] = parent_id

            # 2. 조부모 노드 찾기
            grandparent_mask = (df_gp['start_ts'] <= parent_row['start_ts']) & (df_gp['end_ts'] >= parent_row['end_ts'])
            matching_grandparents = df_gp[grandparent_mask]

            if not matching_grandparents.empty:
                grandparent_row = matching_grandparents.sort_values(by='start_ts', ascending=False).iloc[0]
                seq['grandparent_id'] = grandparent_row['index']

print("매핑 완료.")

# ==============================================================================
# ## 섹션 5: 최종 결과 출력
# ==============================================================================
print("\n" + "="*50)
print("          최종 추출 결과 (L:1 -> L:1 시퀀스)")
print("="*50)

if not l1_sequences:
    print("결과 없음.")
else:
    # 결과를 보기 좋게 DataFrame으로 변환
    results_df = pd.DataFrame(l1_sequences)
    
    # parent_id와 grandparent_id에서 중복을 제거하여 시리즈(Series) 생성
    parent_series = results_df['parent_id'].dropna().unique().astype(int)
    grandparent_series = results_df['grandparent_id'].dropna().unique().astype(int)
    
    print(f"\n[🔥 'L:1 -> L:1' 패턴을 포함하는 부모 노드 ID 시리즈]")
    print(f"총 {len(parent_series)}개")
    print(np.sort(parent_series))
    
    print(f"\n[🔥 'L:1 -> L:1' 패턴을 포함하는 조부모 노드 ID 시리즈]")
    print(f"총 {len(grandparent_series)}개")
    print(np.sort(grandparent_series))

    print("\n\n[상세 내역]")
    print(results_df)

print("\n" + "="*50)
print("분석 종료.")