---
title: "QK-Norm 하에서 PV의 기하학적 적합성"
tags: [QK-Norm, PV, 잔차연결, 가우시안커널, barycenter, Pre-RMSNorm]
sources: [report-20260410-230530.md, 2512.01868v4.pdf, 2305.05465v6.pdf, 내 제안.txt]
created: 2026-04-11
updated: 2026-04-11
---

# QK-Norm 하에서 $PV$의 기하학적 적합성

QK-Norm이 적용된 트랜스포머에서 $PV$를 표준 잔차 연결 $H + PVW_O$로 그대로 사용하는 것이 기하학적으로 부적합한지를 검토한다. 이 페이지는 [[커널해석과_평균이동]]의 논의를 전제로 하며, 찬반 양측의 논거를 균형 있게 정리한다.

## 1. 핵심 질문

QK-Norm에 의해 $P$가 가우시안 RBF 커널 가중치가 되면, $PV$는 단순 가중합이 아니라 **커널 barycentric target**으로 재해석된다. 이때 표준 잔차 연결 $H + PV$는 "좌표 + 좌표"가 되어 부적합한가?

## 2. 선결 문제: QK-Norm이 도입하는 추가적 구조는 정확히 무엇인가?

"$P$가 확률 행렬"이라는 사실은 QK-Norm과 무관하게 소프트맥스의 본래 성질이다. 따라서 **QK-Norm 고유의 추가적 변화**가 무엇인지 명확히 해야 한다.

### 표준 어텐션의 경우

정규화 없이 $q_i^\top k_j$를 전개하면:

$$P_{ij} \propto \exp\Bigl(-\frac{\beta}{2}\|q_i - k_j\|^2 + \frac{\beta}{2}\|k_j\|^2\Bigr)$$

$\|q_i\|^2$ 항은 행 정규화에서 소거되지만, $\|k_j\|^2$는 **키별 bias**로 남는다. 즉 표준 어텐션은 "순수 거리 기반 커널"이 아니라 **거리 항 + 키 노름 prior** 이다.

### QK-Norm 어텐션의 경우

$Q, K$가 구면 위에 있으면 $\|k_j\|^2$가 상수가 되어 소거된다:

$$P_{ij} = \frac{\exp\bigl(-\frac{\beta}{2}\|\hat{q}_i - \hat{k}_j\|^2\bigr)}{\sum_\ell \exp\bigl(-\frac{\beta}{2}\|\hat{q}_i - \hat{k}_\ell\|^2\bigr)}$$

따라서 **QK-Norm만이 점수 함수를 고정된 metric kernel로 정규화**한다. 이 구조적 전환이 있어야 Mean-Shift/KDE 해석([[커널해석과_평균이동]])이 성립한다.

## 3. 부적합하다는 쪽의 논거

1. **강한 이상화 아래에서 $PV$는 변위가 아니라 barycenter이다.**
   $H = V$, $W_O = I$, 토큰이 구면 위를 움직인다고 가정하면, $(PH)_i = \sum_j P_{ij} h_j$는 커널 가중 평균이다. [[rigollet2024]]의 Mean-Shift 등식에서 실제 속도장은 이 barycenter 자체가 아니라 그에 대한 리만 그래디언트이다:
   $$\dot{x}_i = P_{x_i}^\perp\Bigl(\sum_j P_{ij} x_j\Bigr) = P_{x_i}^\perp\Bigl(\sum_j P_{ij} x_j - x_i\Bigr)$$

2. **Rigollet의 SA 동역학에 접선사영이 핵심이다.**
   SA 동역학은 $\dot{x}_i = P_{x_i}^\perp(...)$ 형태이며, 투영 없는 $PV$는 이 이상화와 정확히 맞지 않는다.

3. **$PV$는 "모드 좌표"가 아니라 "모드로 향하는 한 스텝의 barycentric target"이다.** 이 구분은 정밀하게 해야 한다.

## 4. 부적합하지 않다는 쪽의 논거

1. **$H + PVW_O$는 유클리드 공간 연산으로 자기정합적이다.**
   "좌표 + 좌표이므로 모순"이라는 표현은 **내재적(manifold) 해석을 전제할 때만** 성립한다. 실제 잔차 스트림은 $\mathbb{R}^d$의 벡터공간이며, 벡터의 덧셈은 항상 잘 정의된다.

2. **QK-Norm은 $Q, K$에만 적용된다.**
   Mean-Shift 해석이 residual stream 전체로 이전되려면 $VW_O$가 현재 상태 $H$와 같은 기하학적 공간의 점이어야 한다. 그러나 실제로는 $V = HW_V$이고 출력은 $W_O$를 거치므로, $PV$는 일반적으로 **다른 좌표계의 평균**이다.

3. **~~후속 정규화가 1차 수준에서 접선사영 역할을 한다.~~ (Pre-LN에서 부적용)**
   반지름 $r$ 구면으로의 재정규화 $\mathcal{N}_r(y) = ry/\|y\|$에 대해, $\|x\| = r$이면:
   $$\mathcal{N}_r(x + \eta u) = x + \eta\, P_x^\perp u + O(\eta^2)$$
   이 Taylor 전개는 **$\|x\| = r$을 전제**로 한다. Pre-RMSNorm 구조에서 residual stream $H$는 구면 위에 있지 않으므로($\|H\| \neq r$), 이 근사가 성립하지 않는다. 또한 Pre-LN에서는 RMSNorm이 잔차 덧셈 **이후**가 아니라 **이전**(서브레이어 입력)에 적용되므로, "unprojected residual 추가 후 정규화"라는 구도 자체가 존재하지 않는다. 따라서 이 논거는 **Pre-LN 아키텍처에서 방어 논거로 사용할 수 없다.**
   
   한편, "$PV$가 커널 가중 barycentric target이다"라는 공격 논거는 $P$의 커널 구조(QK-Norm)에만 의존하며 $H$가 구면 위에 있을 필요가 없으므로, Pre-LN에서도 그대로 유효하다. 즉 **방어는 무너지지만 공격은 살아 있다.**

4. **참조 논문들의 전제는 현실과 거리가 있다.**
   [[rigollet2024]]: $Q = K = V = I$, 단일헤드, 구면 제약, 연속시간.
   [[geshkovski2023]]: 일반 $Q, K, V$이지만 normalization 없음, 다중헤드/MLP/LN 생략.
   둘 다 실제 LLM 블록의 직접 등가모형은 아니다.

## 5. Pre-RMSNorm에서의 실제 상황

> [!IMPORTANT]
> Pre-RMSNorm (Pre-LN) 구조에서 residual stream $H$는 **초구면 위에 있지 않다.**

Pre-RMSNorm 구조:

```
H_out = H + Attn(RMSNorm(H))
H_out = H_out + FFN(RMSNorm(H_out))
```

- RMSNorm은 **서브레이어 입력에만** 적용 — $H$ 자체를 구면 위로 제약하지 않음
- Residual stream $H$는 레이어를 거치면서 **노름이 계속 성장** — 자유로운 유클리드 공간
- QK-Norm은 $Q, K$ 공간에만 적용 — 구면 제약은 어텐션 스코어 계산에만 존재

따라서 $H$가 초구면 위에 있다는 가정은 Pre-RMSNorm 구조에서 성립하지 않으며, 접선사영이나 구면 기하에 기반한 논의는 직접 적용할 수 없다.

## 6. 종합 판단

| 조건 | 판단 |
|---|---|
| 강한 이상화 ($H=V$, $W_O=I$, 구면 정규화) | $PV$는 기하학적으로 **비정합적** |
| 실제 QK-Norm 트랜스포머 (Pre-RMSNorm) | **"strict contradiction"이 아니라 "특정 이상화 아래 non-canonical"** |

가장 정확한 표현:

> **"QK-Norm은 $PV$를 단순 weighted sum에서 kernel barycenter로 재해석하게 만들지만, 이것이 곧바로 표준 residual이 잘못되었다는 뜻은 아니다."**

수정안에 대해서는 [[잔차연결_수정안]] 참조.
