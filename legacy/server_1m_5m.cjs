// server.cjs (통합 백엔드 서버)
const express = require('express');
const { createClient } = require('@clickhouse/client');
const path = require('path');
const logger = require('./logger.cjs');

// 1. Express 앱 생성 및 설정
const app = express();
const port = 8202; // 서버는 이 포트 하나만 사용한다

// [핵심] Timeframe에 따라 사용할 테이블을 매핑하는 객체
const timeframeToTableMap = {
  '1m': 'candles',
  '5m': 'candles5m',
  '15m': 'candles15m'
  // 여기에 '1h' 등 다른 테이블도 추가하면 바로 확장 가능
  // '1h': 'candles1h',
};

// 2. ClickHouse 클라이언트 설정
const client = createClient({
  // [수정] ClickHouse 데이터베이스의 실제 주소로 수정!
  url: 'http://localhost:8123',
  database: 'default',
});

// 3. 모든 요청 로깅 미들웨어
app.use((req, res, next) => {
  logger.info(`### 통합 서버 요청 수신: ${req.method} ${req.url} ###`);
  next();
});

// 웹 서버 루트 디렉토리 설정
app.use(express.static(path.join(__dirname)));

// 4. HTML 파일 제공 라우트
app.get('/', (req, res, next) => {
  const filePath = path.join(__dirname, 'combined_chart.html');
  logger.info(`통합 서버 HTML 파일 제공 시도: ${filePath}`);
  res.sendFile(filePath, (err) => {
    if (err) {
      logger.error("통합 서버 HTML 파일 전송 오류:", err);
      next(err);
    } else {
      logger.info(`통합 서버 HTML 파일 제공 성공: ${filePath}`);
    }
  });
});

// 5. [통합] Klines 데이터 API 라우트
app.get('/api/klines', async (req, res, next) => {
  logger.info('>>>>>>> 통합 /api/klines 핸들러 진입! <<<<<<<');

  try {
    const symbol = req.query.symbol || 'BTCUSDT';
    // [수정] 프론트에서 요청한 timeframe을 그대로 사용. 기본값은 '1m'
    const timeframe = req.query.timeframe || '1m';

    // [핵심] 요청된 timeframe에 맞는 테이블 이름을 동적으로 선택
    const targetTable = timeframeToTableMap[timeframe];
    if (!targetTable) {
      logger.warn(`  지원하지 않는 timeframe 요청: ${timeframe}`);
      return res.status(400).json({ error: `Unsupported timeframe: ${timeframe}` });
    }

    const requestedLimit = parseInt(req.query.limit, 10);
    const startTime = parseInt(req.query.startTime, 10);
    const endTime = parseInt(req.query.endTime, 10);
    const beforeTimestamp = parseInt(req.query.before, 10);

    const defaultLimit = 5000;
    const maxAllowedLimit = 300000;

    let query = '';
    let queryParams = {
      symbol: symbol,
      timeframe: timeframe, // 쿼리 파라미터에는 timeframe이 여전히 필요
    };
    let reverseData = false;

    logger.info(`  요청 파라미터: symbol=${symbol}, timeframe=${timeframe}, limit=${req.query.limit}, startTime=${req.query.startTime}, endTime=${req.query.endTime}, beforeTimestamp=${beforeTimestamp}`);
    logger.info(`  조회 대상 테이블: ${targetTable} (timeframe: ${timeframe}에 따라 결정됨)`);
    // --- 데이터 조회 로직 (기존 로직과 거의 동일) ---
    if (!isNaN(beforeTimestamp) && !isNaN(requestedLimit) && requestedLimit > 0) {
      const limit = Math.min(requestedLimit, maxAllowedLimit);
      logger.info(`  조회 방식: beforeTimestamp (${beforeTimestamp}) 기준 과거 ${limit}개 조회`);
      query = `
SELECT toUnixTimestamp64Milli(timestamp) AS t, open, high, low, close, volume
FROM ${targetTable}
WHERE symbol = {symbol:String} AND timeframe = {timeframe:String} AND toUnixTimestamp64Milli(timestamp) < {beforeTimestamp:UInt64}
ORDER BY timestamp DESC
LIMIT {limit:UInt64}
        `;
      queryParams.beforeTimestamp = beforeTimestamp;
      queryParams.limit = limit;
      reverseData = true;

    } else if (!isNaN(startTime) && !isNaN(endTime) && startTime < endTime) {
      logger.info(`  조회 방식: 시간 범위 (${startTime} - ${endTime}) 조회`);
      // 1분봉 서버의 쿼리 방식을 따름. endTime을 포함하지 않음.
      query = `
SELECT toUnixTimestamp64Milli(timestamp) AS t, open, high, low, close, volume
FROM ${targetTable}
WHERE symbol = {symbol:String} AND timeframe = {timeframe:String} AND toUnixTimestamp64Milli(timestamp) >= {startTime:UInt64} AND toUnixTimestamp64Milli(timestamp) < {endTime:UInt64} + ${calculateCandleIntervalMillis(timeframe)}
ORDER BY timestamp ASC
        `;
      queryParams.startTime = startTime;
      queryParams.endTime = endTime;
      reverseData = false;

    } else {
      const limit = isNaN(requestedLimit) || requestedLimit <= 0 ? defaultLimit : Math.min(requestedLimit, maxAllowedLimit);
      logger.info(`  조회 방식: 최신 ${limit}개 조회`);
      query = `
SELECT toUnixTimestamp64Milli(timestamp) AS t, open, high, low, close, volume
FROM ${targetTable}
WHERE symbol = {symbol:String} AND timeframe = {timeframe:String}
ORDER BY timestamp DESC
LIMIT {limit:UInt64}
        `;
      queryParams.limit = limit;
      reverseData = true;
    }

    if (!query) {
      logger.warn('  유효하지 않은 Klines 요청 파라미터.');
      return res.status(400).json({ error: 'Invalid request parameters' });
    }

    logger.info('  Executing ClickHouse query');
    logger.info(`  Query parameters: ${JSON.stringify(queryParams)}`);

    try {
      const resultSet = await client.query({ query: query, query_params: queryParams, format: 'JSONEachRow' });
      let data = await resultSet.json();

      if (reverseData) {
        data = data.reverse();
      }

      const formattedData = data.map(row => [
        Number(row.t), Number(row.open), Number(row.high), Number(row.low), Number(row.close), Number(row.volume)
      ]);
      logger.info(`  ClickHouse 조회 성공. ${formattedData.length}개 행 반환.`);
      res.json(formattedData);

    } catch (clickhouseError) {
      logger.error('!!! ClickHouse 쿼리 오류 발생 !!!', clickhouseError);
      res.status(500).json({ error: 'Failed to fetch data from ClickHouse', details: 'Internal server error during database query' });
    }

  } catch (error) {
    logger.error('--- /api/klines 핸들러 내부 오류 ---', error);
    next(error);
  }
});

// 헬퍼 함수 (기존과 동일)
function calculateCandleIntervalMillis(tf) {
  if (tf.endsWith('m')) { return parseInt(tf.slice(0, -1), 10) * 60 * 1000; }
  else if (tf.endsWith('h')) { return parseInt(tf.slice(0, -1), 10) * 60 * 60 * 1000; }
  else if (tf.endsWith('d')) { return parseInt(tf.slice(0, -1), 10) * 24 * 60 * 60 * 1000; }
  return 60000;
}

// 6. 전역 오류 처리 미들웨어
app.use((err, req, res, next) => {
  if (res.headersSent) { return next(err); }
  logger.error('***** 전역 오류 처리기 작동 *****', err);
  res.status(err.status || 500).json({ error: '서버 내부 오류 발생', message: err.message });
});

// 7. 서버 시작
app.listen(port, () => {
  logger.info(`통합 백엔드 서버 실행 중: http://localhost:${port}`);
});