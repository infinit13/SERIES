import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- ❗️ 분석할 파일 경로 설정 (여기를 수정하세요) ❗️ ---
FILE_PATH = "1h_analysis_results_5years_robust.parquet"
# --- ❗️ 군집화할 클러스터 개수 설정 (여기를 수정하세요) ❗️ ---
NUM_CLUSTERS = 4

# --- 메인 클러스터링 로직 ---
try:
    # 1. Parquet 파일 로드
    df = pd.read_parquet(FILE_PATH)
    print(f"'{FILE_PATH}' 파일에서 총 {len(df)}개의 데이터를 불러왔습니다.")

    # 2. 피벗 개수 필터링 (새로 추가된 부분)
    # pivot_count가 4에서 10 사이인 데이터만 선택
    df_filtered = df[(df['pivot_count'] >= 4) & (df['pivot_count'] <= 10)].copy()
    
    if df_filtered.empty:
        print("경고: 필터링 조건에 맞는 데이터가 없습니다. 클러스터링을 실행할 수 없습니다.")
    else:
        print(f"필터링 후 남은 데이터: {len(df_filtered)}개")
    
        # 3. 클러스터링에 사용할 특징(feature) 선택
        features = df_filtered[['retracement_score', 'abs_angle_deg', 'pivot_count']].values
        print(f"총 {features.shape[1]}개의 속성으로 군집화를 진행합니다.")
        
        # 4. 데이터 스케일링
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        print("데이터 스케일링 완료.")
    
        # 5. K-means 클러스터링 실행
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init='auto')
        df_filtered['cluster'] = kmeans.fit_predict(scaled_features)
        print(f"K-means 클러스터링 완료. {NUM_CLUSTERS}개의 클러스터를 생성했습니다.")
    
        # 6. 결과 확인 및 저장
        print("\n--- 클러스터별 데이터 개수 ---")
        print(df_filtered['cluster'].value_counts().sort_index())
        
        print("\n--- 클러스터 중심점 (스케일링 전 원본 값) ---")
        cluster_centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(cluster_centers_orig, columns=['retracement_score_mean', 'abs_angle_deg_mean', 'pivot_count_mean'])
        centers_df.index.name = 'cluster'
        print(centers_df)
    
        output_filename = f"kmeans_clusters_results_{NUM_CLUSTERS}_filtered.csv"
        df_filtered.to_csv(output_filename, index=False)
        print(f"\n최종 클러스터링 결과가 '{output_filename}' 파일로 저장되었습니다.")

except FileNotFoundError:
    print(f"오류: 지정된 파일 '{FILE_PATH}'을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
except Exception as e:
    print(f"예상치 못한 오류가 발생했습니다: {e}")