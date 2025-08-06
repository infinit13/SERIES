

1. logger.cjs - 든든한 기록병 📜
설명
이건 winston이라는 전문 라이브러리를 사용해서 만든 '전문 로거(Logger)' 모듈이야. 서버를 돌릴 때 발생하는 모든 일들을 파일(

logs/app.log)과 콘솔에 예쁘게 찍어주는 역할을 하지.  단순한 


console.log랑은 차원이 달라.

핵심 기능:


타임스탬프: 모든 로그에 YYYY-MM-DD HH:mm:ss.SSS 형식으로 시간을 박아줘서 언제 무슨 일이 있었는지 정확히 알 수 있어. 


로그 레벨: 'info', 'error', 'warn' 같은 레벨을 붙여서 로그의 중요도를 구분할 수 있지. 


자동 디렉토리 생성: 로그를 저장할 logs 폴더가 없으면 알아서 만들어주는 똑똑함까지 갖췄다. 


출력 포맷: 콘솔에는 색깔까지 입혀서 보여주고 , 파일에는 스택 트레이스까지 상세하게 기록해서 나중에 버그 잡을 때 피눈물 흘릴 일을 줄여준다. 


결론적으로, 이 녀석은 뒤에서 묵묵히 모든 것을 기록하는 병참 장교 같은 놈이야. 

server_5min.cjs가 이 녀석을 불러다 쓰는 구조지. 

구동 방법
이건 단독으로 실행하는 파일이 아니야. 다른 놈(server_5min.cjs)이 require해서 쓰는 부품이지. 다만, 이 부품을 쓰려면 사전 준비가 필요해.

의존성 설치: 프로젝트 폴더에서 터미널을 열고 아래 명령어를 입력해서 winston 라이브러리를 설치해야 해.

Bash

npm install winston
2. server_5min.cjs - 5분봉 데이터 공급 기지 🏭
설명
이 파일은 5분봉 캔들 데이터 전용 백엔드 서버다. 네가 구축한 ClickHouse DB에서 5분봉 데이터를 꺼내서 프론트엔드(combined_chart.html)나 다른 분석 스크립트(analyzer.py, UMAP.py)에 제공하는 핵심적인 역할을 맡고 있어.

핵심 기능:


Express 서버: Node.js의 express 프레임워크를 사용해서 8202번 포트에서 웹 서버를 돌린다. 


ClickHouse 연동: @clickhouse/client 라이브러리로 네 DB에 접속해서 candles5m 테이블의 데이터를 조회해. 


API 엔드포인트 (/api/klines): 이 서버의 심장이야. 프론트에서 

symbol, limit, startTime, endTime 같은 파라미터를 붙여서 데이터를 요청하면, 거기에 맞춰 DB에서 데이터를 뽑아 JSON 형식으로 응답해주지. 



정적 파일 제공: 루트 경로(/)로 접속하면 combined_chart.html 파일을 띄워주는 역할도 겸하고 있어. 


로깅: 위에서 설명한 logger.cjs를 가져와서 모든 요청과 에러를 꼼꼼하게 기록한다. 


구동 방법
이 서버를 실행해야 네 모든 프로젝트가 제대로 돌아가.

의존성 설치: 프로젝트 폴더에서 터미널 열고 아래 명령어를 입력해.

Bash

npm install express @clickhouse/client
(이미 logger.cjs 때문에 winston을 설치했다면 그건 빼도 돼.)

전제 조건:

ClickHouse 서버가 http://localhost:8123에서 실행 중이어야 해.

ClickHouse 

default 데이터베이스 안에 candles5m 테이블이 존재하고 데이터가 들어있어야겠지. 

logger.cjs와 combined_chart.html 파일이 server_5min.cjs와 같은 폴더에 있어야 한다.

서버 실행: 터미널에서 아래 명령어를 입력하면 서버가 가동된다.

Bash

node server_5min.cjs
"5분봉 백엔드 서버 실행 중: http://localhost:8202" 메시지가 뜨면 성공! 

3. combined_chart.html - 모든 것을 담은 통합 분석 전투 지휘소 🗺️
설명
이건 그냥 HTML 파일이 아니야. 네 분석 시스템의 메인 프론트엔드이자, 인터랙티브 차트 그 자체다. 순수 자바스크립트와 Plotly.js 라이브러리만으로 어지간한 상용 차트 툴 뺨치는 기능을 구현해냈어.

핵심 기능:

통합 UI: 타임프레임 변경, 분석 도구, 그리기 도구, 줌/팬 컨트롤, 과거 데이터 조회(타임머신) 기능까지 상단 컨트롤 패널에 모두 때려 박았어.

데이터 연동: server_5min.cjs에 데이터를 요청하고, 바이낸스 웹소켓에 직접 접속해서 실시간 데이터를 받아와 차트를 업데이트해.

고급 상호작용: 키보드 단축키(c, +, -, 화살표 등)로 차트 조작이 가능하고, 마우스 휠 줌, 클릭/우클릭 메뉴 등 사용자와의 상호작용이 아주 풍부해.

💥 하이브리드 분석 엔진 💥: 이 코드의 정수(精髓)야.

analyzeEntireChart 함수는 차트 전체를 스캔해서 메인 시리즈(굵직한 추세)와 서브 시리즈(추세 사이의 연결 구간)를 자동으로 식별해.

extractSeriesFeatures 함수는 식별된 각 시리즈의 특성(기간, 기울기, 거래량, 피봇 개수 등)을 추출해.

calculateTensor와 calculateRetracementVector 함수는 추출된 특성을 바탕으로 각 시리즈를 141차원의 텐서(벡터)로 변환해버려. 이건 나중에 머신러닝 모델에 넣기 위한 최종 데이터 가공 단계라고 볼 수 있지. 되돌림 비율을 구간별로 원핫 인코딩하는 등 매우 정교한 로직이 담겨 있어.

구동 방법
이 파일은 server_5min.cjs가 서빙해주기 때문에 직접 열 필요 없어.

server_5min.cjs를 실행한다.

웹 브라우저를 열고 주소창에 http://localhost:8202를 입력하면 이 화면이 뜰 거야.

4. analyzer.py & UMAP.py - 패턴 탐사 및 군집 분석 플랫폼 🌌
이 두 파일은 사실상 버전 1과 버전 2의 관계라서 묶어서 설명할게. 둘 다 네가 combined_chart.html의 분석 엔진으로 뽑아낸 방대한 패턴 데이터(.parquet 파일)를 시각적으로 탐색하기 위한 전문 분석 대시보드야.

설명
analyzer.py가 기본형이고, UMAP.py가 UMAP(차원 축소)과 DBSCAN(군집화) 기능을 추가한 완전체 버전이지.

analyzer.py (기본형 대시보드)

기능:

.parquet 파일에 저장된 수많은 패턴(시리즈) 데이터를 불러와.

패턴들을 3D 공간에 점으로 뿌려줘. (X축: 되돌림 점수, Y축: 시각적 각도, Z축: 피봇 개수)

3D 차트에서 특정 점을 클릭하거나 원본 인덱스를 입력하면, 해당 패턴의 상세 차트를 mplfinance를 이용해 스크린샷(PNG 파일)으로 저장해줘. 이 과정에서 server_5min.cjs에 접속해서 원본 캔들 데이터를 다시 받아오는 치밀함도 보이지.

UMAP.py (고급 탐사 플랫폼)

기능: analyzer.py의 모든 기능을 포함하면서, 훨씬 강력한 탐색 기능이 추가됐어.

UMAP 차원 축소: 3차원(혹은 그 이상)의 패턴 특징을 사람이 보기 좋은 2차원 평면에 의미 있게 펼쳐줘. 이걸로 데이터의 숨겨진 구조나 관계를 파악할 수 있지.

DBSCAN 군집화: UMAP으로 펼친 2D 공간이나 원래 3D 공간에서 비슷한 패턴들끼리 자동으로 그룹(클러스터)을 묶어줘. "어떤 종류의 패턴들이 주로 나타나는가?"를 한눈에 볼 수 있게 되는 거야.

고급 UI: 사용자가 직접 UMAP과 DBSCAN의 파라미터(이웃 수, 최소 거리 등)를 조절하며 분석을 실행하고, '3D 원본' 뷰와 '2D UMAP' 뷰를 오갈 수 있어. 점들의 색상도 '방향(UP/DOWN)' 기준 또는 'DBSCAN 클러스터' 기준으로 바꿔볼 수 있어서 탐색의 깊이가 달라.

구동 방법
두 스크립트 모두 Python의 dash 라이브러리로 만들어졌고, 실행 방법은 비슷해. UMAP.py가 상위 호환이니 이걸 기준으로 설명하지.

의존성 설치: 터미널에서 아래 명령어를 입력해. UMAP.py에 필요한 모든 라이브러리가 포함되어 있어.

Bash

pip install dash plotly pandas numpy requests mplfinance pyarrow umap-learn scikit-learn
전제 조건:

**server_5min.cjs**가 http://localhost:8202에서 실행 중이어야 해 (스크린샷 기능 때문).


analysis_results_5years_robust.parquet 파일이 UMAP.py와 같은 폴더에 있어야 한다. 

대시보드 실행: 터미널에서 아래 명령어를 입력해.

Bash

python UMAP.py
"웹 브라우저에서 http://127.0.0.1:8080 주소로 접속하세요." 라는 메시지가 보일 거야. 해당 주소로 접속하면 대시보드를 사용할 수 있다.

