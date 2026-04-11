---
title: "The Emergence of Clusters in Self-Attention Dynamics"
tags: [셀프어텐션, 클러스터링, 상호작용입자계, 동역학, 연속시간]
sources: [2305.05465v6.pdf]
created: 2026-04-10
updated: 2026-04-10
---

# The Emergence of Clusters in Self-Attention Dynamics

- **저자:** Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, Philippe Rigollet (MIT)
- **arXiv:** 2305.05465v6 (2024.02)
- **분류:** cs.LG

## 논문의 목표

트랜스포머의 셀프 어텐션이 생성하는 **표현(representation)의 기하학적 구조**를 수학적으로 규명한다. 토큰을 입자(particle)로, 레이어 통과를 시간 진화로 해석하여 $t \to \infty$에서 토큰들이 어떤 형태로 클러스터링되는지 증명한다.

## 아키텍처 전제

이 논문이 분석하는 대상은 실제 트랜스포머가 아닌 극도로 단순화된 모델이다:

| 전제 | 설명 |
|------|------|
| 순수 셀프 어텐션 | MLP 블록 없음 (Section 12.2에서 수치적으로만 논의) |
| 단일 헤드 | Multi-head 없음 (Section 12에서 open problem) |
| 가중치 시간 독립 | $Q, K, V$가 모든 레이어에서 동일 (ALBERT 스타일 weight sharing) |
| 레이어 정규화 없음 | LayerNorm/RMSNorm 미적용. 수학적 대리물로 리스케일링 사용 |
| 연속 시간 | 이산 레이어를 ODE로 이상화 (Remark 3.4에서 이산 전이 증명) |
| 바이어스 없음 | 어텐션에 bias term 없음 |

## 기본 동역학

### 원래 동역학 (식 1.1)

$$\dot{x}_i(t) = \sum_{j=1}^{n} P_{ij}(t)\, V x_j(t)$$

여기서 어텐션 행렬 $P$는 표준 소프트맥스:

$$P_{ij}(t) = \frac{\exp\langle Qx_i(t),\, Kx_j(t)\rangle}{\sum_{\ell=1}^{n} \exp\langle Qx_i(t),\, Kx_\ell(t)\rangle}$$

$P$는 **행합이 1인 확률적(stochastic) 행렬**이다.

### 이산 시간 대응 (식 1.3)

$$x_i^{[k+1]} = x_i^{[k]} + \Delta t \sum_{j=1}^{n} P_{ij}^{[k]}\, V x_j^{[k]}$$

이것이 잔차 연결(residual connection)을 포함한 실제 트랜스포머 레이어와 대응된다.

### 리스케일 동역학 (식 3.1)

토큰의 노름이 지수적으로 발산하는 것을 제어하기 위해 $z_i(t) = e^{-tV} x_i(t)$로 치환하면:

$$\dot{z}_i(t) = \sum_{j=1}^{n} P_{ij}(t)\, V\bigl(z_j(t) - z_i(t)\bigr)$$

이 리스케일링은 레이어 정규화의 수학적 대리물로 도입되었다. 리스케일된 토큰에 대한 어텐션 행렬은 원래 토큰의 것과 동일하다.

## 주요 결과

### Theorem 2.1 — 셀프 어텐션 행렬의 저랭크 불린 수렴

- **조건:** $d = 1$, $V > 0$, $QK > 0$
- **결론:** $P(t)$가 0과 1로만 구성된 저랭크 불린 행렬 $P^*$로 수렴 (이중지수 속도)
- **의미:** 소수의 "리더" 토큰이 나머지 전체의 어텐션을 지배. 전형적으로 $\text{rank}(P^*) \leq 2$.
- **제한:** $d = 1$에서만 증명. 고차원 확장은 Conjecture 4.3 및 수치 실험으로 뒷받침.

### Theorem 3.1 — 볼록 다면체 꼭짓점으로의 클러스터링

- **조건:** $V = I_d$, $Q^\top K > 0$ (양정치)
- **결론:** 리스케일된 토큰 $z_i(t)$가 어떤 볼록 다면체 $\mathcal{K}$의 경계 $\partial\mathcal{K}$로 수렴. 일반적(generic) 초기 조건에서는 꼭짓점으로 수렴.
- **의미:** 볼록 다면체는 초기 토큰과 $Q^\top K$에 의해 완전히 결정됨 → **문맥 인식(context-awareness)**.

### Theorem 4.2 — 초평면으로의 클러스터링

- **조건:** "Good triple" — (1) $V$의 최대 고유값 $\lambda_1(V)$이 양수이고 단순: $\lambda_1(V) > |\lambda_2(V)|$, (2) $\langle Q\phi_1, K\phi_1 \rangle > 0$ ($\phi_1$은 $\lambda_1$의 고유벡터)
- **결론:** 토큰 $z_i(t)$가 $\phi_1$ 방향으로 최대 **3개의 평행 초평면** 중 하나로 수렴.
- **의미:** $V$의 스펙트럼이 클러스터 기하학을 결정. $\phi_1$ 방향의 성분이 선형 분리 가능한 표현을 생성.
- **실증:** ALBERT-xlarge-v2의 head 5, 14에서 good triple 조건 실제 성립 확인 (Fig. 10).

### Theorem 5.2 — 다면체와 부분공간의 혼합

- **조건:** "Good triple with multiplicity" — (1) $Q^\top K > 0$, (2) $V$ paranormal: $V|_F = \lambda I$ ($\lambda > 0$), $\rho(V|_G) < \lambda$ (직교 분해 $\mathbb{R}^d = F \oplus G$)
- **결론:** $z_i(t) \to (\partial\mathcal{K} \cup \{0\}) \times G$ — $F$ 방향으로는 다면체 경계, $G$ 방향으로는 수렴 없음.
- **의미:** Thm 3.1과 Thm 4.2의 일반화.

## 핵심 메커니즘

1. **볼록 껍질 축소:** 시간이 지남에 따라 토큰들의 볼록 껍질이 단조 감소 (Prop. 8.2). $P$가 확률 행렬이므로 $z_i$의 업데이트가 항상 기존 볼록 껍질 안에 머묾.
2. **리더 출현:** 정보량이 큰 토큰이 다른 토큰들의 어텐션을 흡수. $P$의 저랭크 수렴과 대응.
3. **Skip connection의 역할:** skip connection이 없으면 모든 토큰이 단일 점으로 붕괴 (rank collapse, [DCL21]). skip connection이 있어야 풍부한 다중 클러스터 구조가 발생.

## 열린 문제 및 확장

- **Conjecture 4.3 (Codimension):** $V$의 양의 실수부 고유값이 $k$개이면, 토큰이 여차원 $k$의 부분공간 3개로 수렴할 것.
- **Multi-head 확장:** 여러 헤드의 상호작용 하에서의 클러스터링 — open problem.
- **MLP 포함:** 피드포워드 레이어 추가 시 클러스터링이 유지되는지 — 수치적으로 ReLU/tanh에 따라 다른 양상 관찰 (Fig. 18).
- **$Q^\top K > 0$ 완화:** 수치적으로는 이 조건 없이도 클러스터링 발생 (Fig. 16, 17).

## 관련 연구 위치

- [[Rigollet2024]] — 본 논문의 공저자(Rigollet)가 평균장 극한으로 확장한 후속 연구
- Krause 모델, Cucker-Smale 모델 등 의견 동역학/군집 모델과의 유사성
- Sinkformer ([SABP22]) — 어텐션을 Wasserstein gradient flow로 해석
- [DCL21] — skip connection 없는 순수 어텐션의 rank collapse 증명
