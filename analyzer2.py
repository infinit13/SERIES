
import dash
from dash import dcc, html
from dash import Input, Output, State, ctx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from datetime import datetime
import traceback

# --- 분석 라이브러리 임포트 ---
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
import umap # umap-learn >= 0.5 필요

# ==============================================================================
# ## 섹션 1: 데이터 로드 함수
# ==============================================================================
def get_force_score(df):
    # 'abs_angle_deg'가 없는 경우를 대비하여 기본값 1을 사용
    return df['retracement_score'] * df.get('abs_angle_deg', 1).abs()

def load_analysis_data(parent_path, child_path, kline_path):
    """지정된 파일 경로에서 데이터를 로드하고 전처리합니다."""
    try:
        df_parent = pd.read_parquet(parent_path)
        df_child_full = pd.read_parquet(child_path)
        df_klines_full = pd.read_parquet(kline_path)

        # K-line 데이터에 open, high, low, close가 없으면 합성합니다.
        # 실제 OHLC 데이터가 있다면 이 부분을 비활성화해야 합니다.
        if 'close' not in df_klines_full.columns:
            print("경고: K-line 데이터에 OHLC 정보가 없어 합성합니다.")
            df_klines_full['close'] = df_klines_full['retracement_score'] * df_klines_full.get('abs_angle_deg', 1)
            df_klines_full['open'] = df_klines_full['close'] - df_klines_full['retracement_score']
            df_klines_full['high'] = df_klines_full['close'] + df_klines_full.get('pivot_count', 0)
            df_klines_full['low'] = df_klines_full['close'] - df_klines_full.get('abs_angle_deg', 1)

        df_parent['force_score'] = get_force_score(df_parent)
        df_child_full['force_score'] = get_force_score(df_child_full)

        child_max_force, child_sum_force = [], []
        for _, parent_row in df_parent.iterrows():
            children = df_child_full[(df_child_full['start_ts'] >= parent_row['start_ts']) & (df_child_full['end_ts'] <= parent_row['end_ts'])]
            child_max_force.append(children['force_score'].max() if not children.empty else 0)
            child_sum_force.append(children['force_score'].sum() if not children.empty else 0)
        
        df_parent['child_max_force'], df_parent['child_sum_force'] = child_max_force, child_sum_force
        
        y = pd.Series(-1, index=df_parent.index, dtype=int)
        strong_mask = df_parent['child_max_force'] > 0
        if strong_mask.any():
            scores_to_transform = df_parent.loc[strong_mask, ['child_max_force']]
            n_unique = scores_to_transform.nunique().iloc[0]
            n_q = min(5, n_unique) if n_unique > 0 else 1
            if n_q > 1:
                qt = QuantileTransformer(n_quantiles=n_q, output_distribution='uniform', random_state=42)
                y[strong_mask] = pd.cut(qt.fit_transform(scores_to_transform).flatten(), bins=n_q, labels=False, include_lowest=True)
            else: y[strong_mask] = 0
        df_parent['y_child_strength_label'] = y
        
        # JSON 직렬화를 위해 datetime 객체는 제거하고 to_dict를 호출합니다.
        return {
            'parent_data': df_parent.to_dict('records'), 
            'child_data_full': df_child_full.to_dict('records'), 
            'klines_full': df_klines_full.to_dict('records')
        }
    except Exception as e:
        traceback.print_exc()
        return None

# ==============================================================================
# ## 섹션 2: Dash 앱 레이아웃
# ==============================================================================
app = dash.Dash(__name__)
app.title = "가중 UMAP 탐색적 패턴 분석"

analysis_data_store = load_analysis_data(
    '15m_analysis_results_5years_robust_BTCUSDT.parquet',
    '5m_analysis_results_5years_robust_BTCUSDT.parquet',
    '1m_analysis_results_5years_robust_BTCUSDT.parquet'
)

app.layout = html.Div(style={'backgroundColor': '#1E1E1E', 'color': '#E0E0E0', 'fontFamily': 'sans-serif'}, children=[
    dcc.Store(id='analysis-data-store', data=analysis_data_store),
    html.H1(children='가중 UMAP 탐색적 패턴 분석 플랫폼', style={'textAlign': 'center', 'padding': '15px'}),
    
    # --- 필터링 및 설정 UI (기존과 동일) ---
    html.Div([
         html.Div([
            html.H4("1. 데이터 필터링", style={'marginTop': '0', 'marginBottom': '10px'}),
            html.Div([
                html.Div([html.Label("되돌림 점수:"), dcc.Input(id='ret-score-min', type='number', placeholder='최소'), dcc.Input(id='ret-score-max', type='number', placeholder='최대')], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([html.Label("피봇 개수:"), dcc.Input(id='pivot-min', type='number', placeholder='최소'), dcc.Input(id='pivot-max', type='number', placeholder='최대')], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([html.Label("절대 각도:"), dcc.Input(id='slope-min', type='number', placeholder='최소'), dcc.Input(id='slope-max', type='number', placeholder='최대')], style={'display': 'inline-block', 'padding': '5px 10px'}),
                html.Div([html.Label("방향:"), dcc.Dropdown(id='direction-dropdown', options=[{'label': '전체', 'value': 'all'}, {'label': 'UP', 'value': 'up'}, {'label': 'DOWN', 'value': 'down'}], value='all', clearable=False)], style={'display': 'inline-block', 'width': '150px', 'padding': '5px 10px', 'verticalAlign': 'middle'}),
            ], style={'textAlign': 'center'}),
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px', 'marginBottom': '10px'}),

        html.Div([
            html.H4("2. 분석 및 시각화 설정", style={'marginTop': '0', 'marginBottom': '10px'}),
            html.Div([
                html.Label("Gamma (자식 세력 강조 강도):"),
                dcc.Slider(id='gamma-slider', min=0, max=1, step=0.05, value=0.25, marks={i/10:str(i/10) for i in range(11)})
            ], style={'width': '50%', 'margin': 'auto', 'padding': '10px'}),
            html.Div([
                dcc.Dropdown(id='color-selector', options=[{'label': '채색: 자식 세력 등급', 'value': 'child_strength'}, {'label': '채색: 자식 총점수', 'value': 'child_sum_force'}, {'label': '채색: 부모 세력 점수', 'value': 'force_score'}, {'label': '채색: 방향 (UP/DOWN)', 'value': 'direction'}], value='child_strength', clearable=False, style={'width': '250px', 'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '10px'}),
                html.Button('분석 및 시각화 실행', id='run-button', n_clicks=0, style={'padding': '8px 15px'}),
            ], style={'textAlign': 'center', 'marginTop': '10px'}),
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px'}),
    ], style={'textAlign': 'center', 'padding': '10px'}),

    dcc.Loading(id="loading-icon", children=[dcc.Graph(id='main-chart', style={'height': '60vh'})], type="default"),
    # --- 스냅샷을 표시할 캔들차트 영역 ---
    dcc.Loading(dcc.Graph(id='child-overview-chart', style={'height': '50vh'})),
    html.Div(id='click-output', style={'textAlign': 'center', 'padding': '10px', 'fontSize': '16px', 'fontWeight': 'bold'}),
])

# ==============================================================================
# ## 섹션 3: Dash 콜백
# ==============================================================================
@app.callback(
    Output('main-chart', 'figure'), 
    Output('child-overview-chart', 'figure'), 
    Output('click-output', 'children'),
    Input('run-button', 'n_clicks'), 
    Input('main-chart', 'clickData'),
    State('analysis-data-store', 'data'),
    State('ret-score-min', 'value'), State('ret-score-max', 'value'), State('pivot-min', 'value'), State('pivot-max', 'value'),
    State('slope-min', 'value'), State('slope-max', 'value'), State('direction-dropdown', 'value'),
    State('color-selector', 'value'), State('gamma-slider', 'value'),
    prevent_initial_call=True
)
def universal_callback(run_clicks, clickData, analysis_data, rs_min, rs_max, p_min, p_max, s_min, s_max, direction,
                       color_mode, gamma):
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'
    empty_fig = go.Figure().update_layout(template='plotly_dark', xaxis={'visible': False}, yaxis={'visible': False})

    if not analysis_data:
        return go.Figure().update_layout(title_text='데이터 로드 실패'), empty_fig, "데이터 로드 실패"
               
    df_parent_full = pd.DataFrame(analysis_data['parent_data'])

    if triggered_id == 'main-chart' and clickData:
        # 1. 필요한 데이터프레임 로드 및 시간 인덱스 설정
        df_child_full = pd.DataFrame(analysis_data['child_data_full'])
        df_klines_full = pd.DataFrame(analysis_data['klines_full'])
        df_klines_full['timestamp'] = pd.to_datetime(df_klines_full['start_ts'], unit='ms')
        df_klines_full.set_index('timestamp', inplace=True)
        
        # 2. 클릭된 부모 노드 정보 가져오기
        pid = clickData['points'][0]['customdata']
        P = df_parent_full.loc[pid]
        P_start_dt = pd.to_datetime(P['start_ts'], unit='ms')
        P_end_dt = pd.to_datetime(P['end_ts'], unit='ms')

        # 3. 스냅샷을 위한 시간 범위 설정 (패딩 적용)
        padding = (P['end_ts'] - P['start_ts']) * 1.5
        start_dt_padded = pd.to_datetime(int(P['start_ts'] - padding), unit='ms')
        end_dt_padded = pd.to_datetime(int(P['end_ts'] + padding), unit='ms')
        
        # 4. 상세 K-line 데이터 필터링 및 차트 생성
        klines_snapshot = df_klines_full[(df_klines_full.index >= start_dt_padded) & (df_klines_full.index <= end_dt_padded)]
        
        if klines_snapshot.empty:
            msg = f"오류: 부모 ID #{pid}에 대한 상세 K-line 데이터를 찾을 수 없습니다."
            return dash.no_update, empty_fig.update_layout(title=msg), msg

        fig_snapshot = go.Figure()
        fig_snapshot.add_trace(go.Candlestick(
            x=klines_snapshot.index,
            open=klines_snapshot['open'], high=klines_snapshot['high'],
            low=klines_snapshot['low'], close=klines_snapshot['close'],
            name='1m K-lines',
            increasing_line_color='rgba(100,200,100,0.8)', 
            decreasing_line_color='rgba(230,100,100,0.8)'
        ))
        
        # 5. 부모 노드 영역을 반투명한 사각형으로 표시
        fig_snapshot.add_vrect(
            x0=P_start_dt, x1=P_end_dt,
            fillcolor="yellow", opacity=0.15,
            layer="below", line_width=1, line_color="yellow",
            annotation_text=f"Parent (ID:{pid})", annotation_position="top left"
        )

        # 6. 자식 노드들을 각각의 영역(사각형)으로 표시
        kids = df_child_full[(df_child_full['start_ts'] >= P['start_ts']) & (df_child_full['end_ts'] <= P['end_ts'])]
        
        if not kids.empty:
            for _, k_row in kids.iterrows():
                kid_start_dt = pd.to_datetime(k_row['start_ts'], unit='ms')
                kid_end_dt = pd.to_datetime(k_row['end_ts'], unit='ms')
                color = "green" if k_row['direction'] == 1.0 else "red"
                fig_snapshot.add_vrect(
                    x0=kid_start_dt, x1=kid_end_dt,
                    fillcolor=color, opacity=0.4,
                    layer="below", line_width=1, line_color="white"
                )

        # 7. 차트 레이아웃 업데이트
        fig_snapshot.update_layout(
            template='plotly_dark',
            title=f"부모 노드(ID: {pid}) 상세 스냅샷 및 자식 노드({len(kids)}개) 위치",
            xaxis_rangeslider_visible=False,
            margin=dict(l=40, r=20, b=30, t=60)
        )
        
        msg = f"부모 노드 #{pid}의 스냅샷과 자식 노드({len(kids)}개)를 표시했습니다."
        return dash.no_update, fig_snapshot, msg

    elif triggered_id == 'run-button':
        # (필터링 및 UMAP 생성 로직)
        query_parts = []
        if rs_min is not None: query_parts.append(f"retracement_score >= {rs_min}")
        if rs_max is not None: query_parts.append(f"retracement_score <= {rs_max}")
        if p_min is not None: query_parts.append(f"pivot_count >= {p_min}")
        if p_max is not None: query_parts.append(f"pivot_count <= {p_max}")
        if s_min is not None: query_parts.append(f"abs_angle_deg >= {s_min}")
        if s_max is not None: query_parts.append(f"abs_angle_deg <= {s_max}")
        if direction != 'all': query_parts.append(f"direction == {1.0 if direction == 'up' else -1.0}")
        query_str = " and ".join(query_parts)
        
        df_filtered = df_parent_full.query(query_str) if query_parts else df_parent_full
        
        if len(df_filtered) < 2:
            msg = "오류: 필터링된 데이터가 너무 적어 UMAP을 생성할 수 없습니다."
            return go.Figure().update_layout(title_text=msg, template='plotly_dark'), empty_fig, msg

        features = df_filtered[['retracement_score', 'abs_angle_deg', 'pivot_count']].values
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, target_weight=gamma)
        y_labels = df_filtered['y_child_strength_label'].values if gamma > 0 else None
        embedding = reducer.fit_transform(StandardScaler().fit_transform(features), y=y_labels)
        
        # (채색 및 호버 텍스트 로직)
        hover_texts = [f"인덱스: {idx}<br>부모점수: {r['force_score']:.2f}<br>자식총점: {r['child_sum_force']:.2f}" for idx, r in df_filtered.iterrows()]
        marker_color, cbar_title = df_filtered['y_child_strength_label'], '자식 세력 등급'

        fig_umap = go.Figure(data=[go.Scattergl(x=embedding[:, 0], y=embedding[:, 1], mode='markers', customdata=df_filtered.index,
                                          marker=dict(size=7, color=marker_color, colorscale='Plasma', showscale=True, colorbar={'title': cbar_title}),
                                          text=hover_texts, hoverinfo='text')])
        fig_umap.update_layout(template='plotly_dark', title_text=f'2D UMAP 사영 | 데이터: {len(df_filtered)}개, Gamma: {gamma}')
        
        msg = f"분석 완료. 필터링된 데이터: {len(df_filtered)}개"
        return fig_umap, empty_fig.update_layout(title="UMAP에서 부모 노드를 클릭하면 여기에 상세 차트가 표시됩니다."), msg

    # 앱 초기 로딩 시
    return go.Figure().update_layout(template='plotly_dark', title="[분석 및 시각화 실행] 버튼을 눌러주세요."), \
           empty_fig.update_layout(template='plotly_dark'), ""


if __name__ == '__main__':
    app.run(debug=True, port=8057)