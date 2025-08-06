// logger.js
const winston = require('winston');
const path = require('path');
const fs = require('fs');

// 로그 파일을 저장할 디렉토리 설정
const logDir = 'logs';
const logDirectory = path.join(__dirname, logDir);

// 로그 디렉토리 생성 (없으면)
if (!fs.existsSync(logDirectory)) {
  try {
    fs.mkdirSync(logDirectory);
  } catch (err) {
    // 디렉토리 생성 실패 시 콘솔에 오류 출력 (로거 초기화 전이므로 console 사용)
    console.error('로그 디렉토리 생성 실패:', err);
    // 필요시 프로세스 종료 또는 다른 오류 처리
    process.exit(1);
  }
}

const logFilePath = path.join(logDirectory, 'app.log');

// 로거 생성
const logger = winston.createLogger({
  level: 'info', // 로그 레벨 (info 이상만 기록)
  format: winston.format.combine(
    winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss.SSS' }), // 타임스탬프 형식
    winston.format.errors({ stack: true }), // 오류 스택 트레이스 포함
    winston.format.splat(),
    winston.format.printf(({ timestamp, level, message, stack }) => { // 파일 로그 포맷
      // 스택 트레이스가 있으면 포함
      return `${timestamp} [${level.toUpperCase()}]: ${stack || message}`;
    })
  ),
  defaultMeta: { service: 'backend-server' }, // 기본 메타데이터 (선택 사항)
  transports: [
    // 콘솔 출력 설정
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(), // 콘솔에는 색상 적용
        winston.format.printf(({ timestamp, level, message, stack }) => { // 콘솔 로그 포맷
          return `${timestamp} [${level}]: ${stack || message}`;
        })
      )
    }),
    // 파일 출력 설정
    new winston.transports.File({ filename: logFilePath })
  ]
});

// 생성된 로거 인스턴스를 내보내기
module.exports = logger;