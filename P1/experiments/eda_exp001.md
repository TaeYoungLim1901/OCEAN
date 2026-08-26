# exp001 EDA (2026-08-13)

- 스키마/행수: train 776,706 × 9, test 169,011 × 7. `temp` 결측 0, train 양성 32,126(4.1362%).
- 양성 행: noise 9,442, drift 8,575, offset 7,296, flatline 6,030, spike 101, 나머지는 복합 유형.
- label/type/gap 기준 양성 event 289개. 단일 유형 event 수/중앙 지속시간: spike 101/0.17h, flatline 55/14.33h, noise 53/32.50h, offset 35/30.67h, drift 32/51.08h.
- 정상 대비 유형 신호 중앙값: 정상 `|center spike|=0.0201`, noise 1.5288, spike 6.4842; flatline의 이전 값과 정확히 동일한 비율 99.10%.
- 모든 stream은 10분 격자가 우세하지만 gap이 있으므로 gap을 넘는 diff/rolling/event 연결을 금지했다.
- `baseline_rule.csv`: 양성 4,402(2.6046%), 양성 run 1,199개 중 singleton 699개, 3시간 미만 1,164개, 최장 38.67h. 숨은 test 정답이 없어 실패를 직접 채점할 수는 없지만, 문제의 지속형 이상이 최대 86.5h이고 train 중앙 지속시간도 30~51h인 것과 비교하면 baseline은 짧은 후보에 집중된 형태다.
- 검증: station-layer별 4개 순차 block, 검증 block 주변 87h 학습 purge. test가 다음 해이므로 이 OOF는 최종 판정기가 아니라 탐색기다.
