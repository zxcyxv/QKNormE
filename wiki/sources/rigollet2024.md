---
title: "The Mean-Field Dynamics of Transformers"
tags: [평균장동역학, 초구면, 클러스터링, 쿠라모토, 가우시안커널, 정규화]
sources: [2512.01868v4.pdf]
created: 2026-04-10
updated: 2026-04-10
---

# The Mean-Field Dynamics of Transformers

- **저자:** Philippe Rigollet (MIT)
- **arXiv:** 2512.01868v4 (2024.12)
- **분류:** math.PR / cs.LG

## 논문의 목표

트랜스포머 어텐션을 **초구면 $S^{d-1}$ 위의 상호작용 입자계**로 이상화하여, 평균장(mean-field) 극한에서의 클러스터링 동역학을 엄밀하게 분석한다. Wasserstein gradient flow, 쿠라모토 동기화 모델, Mean-Shift 클러스터링과의 연결을 확립한다.

## 아키텍처 전제

| 전제 | 설명 |
|------|------|
| **토큰 공간: $S^{d-1}$** | 토큰이 단위 초구면 위에 존재. 레이어 정규화를 구면으로의 사영 $P^\perp_x$로 이상화 |
| **$Q = K = V = I$** | 쿼리·키·밸류 행렬이 모두 항등행렬. 어텐션 스코어가 순수 내적 $\langle x_i, x_j \rangle$ |
| **명시적 역온도 $\beta > 0$** | 어텐션 스코어에 $\beta$를 곱함. 실제 트랜스포머의 $1/\sqrt{d}$ 스케일링에 대응 |
| **순수 셀프 어텐션** | MLP 없음 |
| **단일 헤드** | Multi-head 없음 |
| **연속 시간** | 레이어를 연속 시간 ODE로 이상화 |

### Geshkovski (2023)와의 핵심 차이

| | Geshkovski (2023) | Rigollet (2024) |
|---|---|---|
| 토큰 공간 | $\mathbb{R}^d$ (유클리드) | $S^{d-1}$ (단위 초구면) |
| 레이어 정규화 | 없음 (리스케일링 $e^{-tV}$로 대체) | **구면 사영 $P^\perp_x$로 이상화** |
| $Q, K, V$ | 일반 행렬 (good triple 조건) | $Q = K = V = I$ |
| 커널 성격 | 내적 커널 (unbounded) | **가우시안 커널** (구면 위 $\langle x_i, x_j \rangle$) |
| 클러스터 결과 | 다중 클러스터 (초평면/다면체) | **궁극적으로 단일 클러스터** (다중은 준안정) |

## 핵심 동역학

### SA (Self-Attention) 동역학

$$\dot{x}_i(t) = P^\perp_{x_i(t)}\!\left(\frac{\sum_{j=1}^{n} e^{\beta\langle x_i, x_j\rangle} x_j}{\sum_{k=1}^{n} e^{\beta\langle x_i, x_k\rangle}}\right)$$

$P^\perp_x y = y - \langle x, y \rangle x$: 접선 평면으로의 직교 사영. 토큰을 $S^{d-1}$ 위에 유지.

### USA (Unnormalized SA) 동역학

소프트맥스 정규화 대신 $1/n$을 사용. 에너지 범함수

$$E_\beta(\mu) = \frac{1}{2\beta}\iint e^{\beta\langle x, y\rangle}\, d\mu(x)\, d\mu(y)$$

에 대한 **Wasserstein gradient flow**로 해석 가능.

## 주요 결과

### 1. 전역 클러스터링 (Theorem 1)

$d \geq 3$, 임의의 $\beta \geq 0$에서, **거의 모든 초기 조건**에 대해 모든 토큰이 단일 점 $x_\infty \in S^{d-1}$으로 수렴:
$$\lim_{t \to \infty} \|x_i(t) - x_j(t)\| = 0 \quad \forall\, i, j$$

다중 클러스터 배치는 새들 포인트 → measure zero 집합 위에서만 안정.

### 2. 지수적 수렴 속도 (Theorem 3)

초기 토큰이 같은 열린 반구에 있으면:
$$\|x_i(t) - x^*\| \leq C e^{-\lambda t}$$

### 3. 준안정성과 다중 클러스터의 느린 합병 (Section 5)

- 다중 클러스터는 **에너지 경관의 새들 포인트** 근방에 머묾
- 준안정 시간 $\log T_2 \sim \beta$ — $\beta$가 클수록 지수적으로 오래 유지
- 에너지의 계단형(staircase) 프로파일: 긴 정체 → 급격한 합병 → 정체 반복
- 가장 가까운 클러스터 쌍이 먼저 합병 (Theorem 6)

### 4. Mean-Shift와의 동치 (Section 5.3)

구면 위에서 가우시안 커널 $K(x) = e^{-\frac{\beta}{2}\|x\|^2}$를 사용한 **blurring Mean-Shift** 알고리즘:

$$\dot{x}_i(t) = \nabla \log (K * \mu_t)(x_i(t))$$

이것이 구면 위에서 항등식 $\|x_i - x_j\|^2 = 2 - 2\langle x_i, x_j \rangle$을 적용하면 **SA 동역학과 정확히 일치**한다.

→ [[커널해석과_평균이동]]에서 상세 분석

### 5. 등각(equiangular) 모델과 정규화 비교 (Section 6)

$\langle x_i, x_j \rangle = \rho$ (모든 $i \neq j$)인 대칭 초기 조건에서 1차원 ODE로 환원:

| 정규화 방식 | 수렴 속도 | 비고 |
|-------------|-----------|------|
| **Post-LN (SA)** | $1 - \rho(t) \sim e^{-2t}$ | 지수적 — 빠른 collapse |
| **Pre-LN** | $1 - \rho(t) \sim 1/t^2$ | 다항식 — collapse 지연에 유리 |
| **USA** | $1 - \rho(t) \sim e^{-2e^\beta t}$ | 이중지수 — 가장 빠른 collapse |

### 6. 긴 문맥 상전이 (Theorem 7)

$\beta_n = \gamma \log n$으로 스케일링 시:

| 영역 | 동작 |
|------|------|
| $\gamma < \frac{1}{1-\rho}$ | 균일 혼합 → collapse |
| $\gamma = \frac{1}{1-\rho}$ | 임계점 — 내용 적응적 희소성 유지 |
| $\gamma > \frac{1}{1-\rho}$ | 항등 작용 — 어텐션 억제 |

### 7. 노이즈 트랜스포머 (Section 7)

$\sqrt{2\kappa^{-1}}\, dW_i(t)$ 항 추가 → McKean-Vlasov SDE. $\kappa \to \infty$에서 결정론적 동역학 복원. 정상 분포의 분기(bifurcation) 구조는 대부분 open problem.

## 관련 연구 위치

- [[geshkovski2023]] — 본 논문의 선행 연구. 동일 저자(Rigollet) 참여. $\mathbb{R}^d$에서의 클러스터링.
- 쿠라모토 모델 ($d=2$, $\beta=0$일 때 SA가 쿠라모토와 일치)
- nGPT [LHSG25] — 초구면 위 표현 학습. 본 논문의 전제와 직접 대응.
- Qwen, SSMax, SWAN-GPT — $\beta \sim \log n$ 스케일링의 실용적 적용
