import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def create_force_deployment_heatmap(df_parent, df_child, sort_parents_by='parent_force_score', max_children_to_show=50, max_parents_to_show=25):
    """
    부모-자식 관계를 기반으로 '세력 전개도' 히트맵을 생성합니다.
    Y축에 부모 인덱스를 표시하는 기능이 추가되었습니다.

    Args:
        df_parent (pd.DataFrame): 부모 데이터 (15분봉 분석 결과)
        df_child (pd.DataFrame): 자식 데이터 (5분봉 분석 결과)
        sort_parents_by (str): 부모(Y축) 정렬 기준. 
                               'parent_force_score', 'child_count', 'parent_start_ts' 중 선택.
        max_children_to_show (int): 히트맵에 표시할 최대 자식 수 (가로축 크기 제한).
        max_parents_to_show (int): 히트맵에 표시할 최대 부모 수 (세로축 크기 제한). 글자 표기를 위해 줄이는 것을 권장.
    """
    print("1. 부모-자식 관계 매핑 시작...")
    
    # 부모 데이터에 고유 ID와 세력 점수 추가
    df_parent = df_parent.copy()
    df_parent['parent_id'] = df_parent.index
    df_parent['force_score'] = df_parent['retracement_score'] * df_parent['abs_angle_deg']
    
    # 자식 데이터에 세력 점수 추가
    df_child = df_child.copy()
    df_child['force_score'] = df_child['retracement_score'] * df_child['abs_angle_deg']

    # 시간 기반으로 부모-자식 매칭
    child_parent_map = []
    for p_idx, parent in df_parent.iterrows():
        children_in_range = df_child[
            (df_child['start_ts'] >= parent['start_ts']) & 
            (df_child['end_ts'] <= parent['end_ts'])
        ]
        for c_idx, child in children_in_range.iterrows():
            child_parent_map.append({
                'parent_id': parent['parent_id'],
                'child_start_ts': child['start_ts'],
                'child_force_score': child['force_score']
            })
    
    if not child_parent_map:
        print("경고: 매칭되는 부모-자식 관계가 없습니다. 히트맵을 생성할 수 없습니다.")
        return

    df_map = pd.DataFrame(child_parent_map)
    print(f"--> {len(df_map)}개의 부모-자식 관계 매핑 완료.")

    print("2. 자식 서열화 및 데이터 피벗 테이블 생성...")
    df_map['child_rank'] = df_map.groupby('parent_id')['child_start_ts'].rank(method='first').astype(int)
    heatmap_pivot = df_map.pivot_table(
        index='parent_id', 
        columns='child_rank', 
        values='child_force_score'
    )
    
    parent_info = df_parent[['parent_id', 'force_score', 'start_ts']].set_index('parent_id')
    heatmap_pivot = parent_info.join(heatmap_pivot, how='right')
    heatmap_pivot['child_count'] = heatmap_pivot.iloc[:, 3:].notna().sum(axis=1)

    print("3. 정렬 옵션 적용...")
    if sort_parents_by == 'parent_force_score':
        sort_col = 'force_score'
        ascending = False
    elif sort_parents_by == 'child_count':
        sort_col = 'child_count'
        ascending = False
    else: # 'parent_start_ts'
        sort_col = 'start_ts'
        ascending = True
        
    sorted_pivot = heatmap_pivot.sort_values(by=sort_col, ascending=ascending)
    sorted_pivot = sorted_pivot.dropna(subset=[sort_col]) # 정렬 기준값이 없는 행 제거
    
    # 표시할 데이터 양 제한
    sorted_pivot = sorted_pivot.head(max_parents_to_show)
    child_columns = sorted_pivot.columns[3:-1]
    
    if len(child_columns) > max_children_to_show:
        child_columns = child_columns[:max_children_to_show]

    parent_scores = sorted_pivot[['force_score']]
    child_scores_heatmap = sorted_pivot[child_columns]

    print("4. 히트맵 시각화 (Y축 인덱스 포함)...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 14))
    
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.5, 10], wspace=0.05)
    
    ax_parent = plt.subplot(gs[0])
    ax_child = plt.subplot(gs[1])

    # 부모 점수 히트맵 (Y축 레이블 없음)
    sns.heatmap(parent_scores, ax=ax_parent, cmap="coolwarm", cbar=True, annot=False, yticklabels=False, cbar_kws={'label': 'Parent Force Score'})
    ax_parent.set_ylabel(f'Parents (Sorted by {sort_parents_by})', fontsize=12)
    ax_parent.set_xlabel('')
    ax_parent.tick_params(axis='x', bottom=False)

    # 자식 점수 히트맵 (Y축 레이블 있음)
    # yticklabels에 정렬된 데이터의 인덱스(parent_id)를 직접 전달
    sns.heatmap(child_scores_heatmap, ax=ax_child, cmap="viridis", cbar=True, annot=False, 
                yticklabels=sorted_pivot.index.astype(int), cbar_kws={'label': 'Child Force Score'})
    ax_child.set_ylabel('')
    ax_child.set_xlabel('Child Sequence (Sorted by Time)', fontsize=12)
    ax_child.tick_params(axis='y', rotation=0, labelsize=8) # Y축 폰트 크기 조절
    
    fig.suptitle('Force Deployment Heatmap: Parent vs. Child Sequence', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_filename = 'force_deployment_heatmap_with_indices.png'
    plt.savefig(output_filename, dpi=150)
    print(f"\n성공! 히트맵을 '{output_filename}' 파일로 저장했습니다.")


# --- 메인 실행 로직 ---
try:
    print("데이터 파일 로딩 중...")
    df_parent_main = pd.read_parquet('15m_analysis_results_5years_robust.parquet')
    df_child_main = pd.read_parquet('analysis_results_5years_robust.parquet')
    print("파일 로딩 완료.")

    create_force_deployment_heatmap(
        df_parent=df_parent_main, 
        df_child=df_child_main,
        sort_parents_by='parent_force_score',
        max_parents_to_show=25 # Y축 레이블을 위해 표시할 부모 수를 줄임
    )
except FileNotFoundError:
    print("오류: '15m_analysis_results_5years_robust.parquet' 또는 'analysis_results_5years_robust.parquet' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"오류 발생: {e}")