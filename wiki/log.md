---
title: 작업 이력
updated: 2026-04-11
---

# 작업 로그

## [2026-04-10] init | 위키 초기화
- 디렉토리 구조 생성 (`raw/`, `wiki/sources/`, `wiki/concepts/`, `wiki/entities/`)
- 스키마 파일 `CLAUDE.md` 작성
- `index.md`, `log.md` 생성
- 원본 자료 3개를 `raw/`로 이동: `2305.05465v6.pdf`, `2512.01868v4.pdf`, `내 제안.txt`

## [2026-04-10] ingest | Geshkovski et al. (2305.05465)
- 논문 전문 읽기 (55페이지)
- `wiki/sources/geshkovski2023.md` 작성 — 아키텍처 전제, 동역학, 4개 주요 정리, 열린 문제 정리
- `wiki/sources/geshkovski2023_appendix_PIV.md` 작성 — 리스케일 동역학에서 $(P-I)V$ 도출 경위 분리 기록
- `wiki/index.md` 업데이트

## [2026-04-10] ingest | Rigollet (2512.01868)
- 논문 전문 읽기 (20페이지)
- `wiki/sources/rigollet2024.md` 작성 — 아키텍처 전제 (초구면, $Q=K=V=I$), SA/USA 동역학, 전역 클러스터링, 준안정성, 등각 모델, 긴 문맥 상전이
- `wiki/concepts/커널해석과_평균이동.md` 작성 — 사용자 제안의 핵심 논리 체인 정리: QK-Norm → 가우시안 커널 → Mean-Shift 변위 벡터 → $(P-I)V$ → 그래프 라플라시안 확산
- `wiki/index.md` 업데이트

## [2026-04-10] ingest | 사용자 제안 (내 제안.txt)
- `wiki/sources/제안_PIV.md` 작성 — 3단계 논증 (커널 전환 → $PV$ 의미 전환 → 잔차 연결 모순과 $(P-I)V$ 도출), 두 논문과의 관계, 미해결 사항 5개 식별
- `wiki/index.md` 업데이트

## [2026-04-11] integrate | ChatGPT 보고서 (report-20260410-230530) 위키 통합
- 보고서 검토: $PV$의 기하학적 적합성, $(P-I)V$의 찬반, 대안 비교 (3개 질문)
- 추가 분석: Pre-RMSNorm에서 $H$는 유클리드 공간 → 초구면 가정 불필요 → $PVW_O - H$가 가장 자연스러운 수정안
- `wiki/concepts/커널해석과_평균이동.md` 수정 — "모순"→"비정합성", $(P-I)V$ 한계 섹션 추가, 위키링크 추가
- `wiki/concepts/PV의_기하학적_적합성.md` 신규 — 보고서 Q1 내용 정리 (찬반 균형, Pre-RMSNorm 분석)
- `wiki/concepts/잔차연결_수정안.md` 신규 — 보고서 Q2+Q3 통합, 6개 대안 비교표, 최종 권고
- `wiki/index.md` 업데이트
