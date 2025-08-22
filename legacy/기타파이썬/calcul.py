import numpy as np

data = {
    "start_ts":1704636000000.0,
    "end_ts":1704681900000.0,
    "retracement_score":1.0,
    "pivot_count":4.0,
    "abs_angle_deg":63.21,
    "direction":-1.0,
    "force_score":63.21,
    "parent_id":-1.0,
    "parent_force_score":0,
    "parent_direction":0,
    "grandparent_id":-1.0,
    "grandparent_force_score":0,
    "grandparent_direction":0
}

# 부모, 조모, 자기 자신 force_score 추출
self_score = data["force_score"]
parent_score = data["parent_force_score"]
grandparent_score = data["grandparent_force_score"]

# 가중치 설정
w_self = 0.1
w_parent = 0.6
w_grandparent = 0.3

# 가중치 적용한 벡터
scores = np.array([
    w_self * self_score,
    w_parent * parent_score,
    w_grandparent * grandparent_score
])

# 소프트맥스 계산
exp_scores = np.exp(scores - np.max(scores))  # 안정화 처리
softmax_values = exp_scores / exp_scores.sum()

print("가중치 적용 소프트맥스 결과:", softmax_values)
