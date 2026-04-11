---
title: "사용자 제안: QK-Norm 하에서 (P-I)V 어텐션 출력"
tags: [QK-Norm, 그래프라플라시안, 확산방정식, 잔차연결, 접선사영, 아키텍처제안]
sources: [내 제안.txt]
created: 2026-04-10
updated: 2026-04-10
---

# 사용자 제안: QK-Norm 하에서 $(P-I)V$ 어텐션 출력

## 핵심 주장 (보수적 프레임)

**단 하나의 핵심 명제:**

> QK-Normalization이 적용되는 트랜스포머에서, 어텐션 출력 $PV$를 잔차 연결에 그대로 더하는 것은 수학적 모순이다.

이 모순을 해소하는 **가장 간단한 변형**이 $(P-I)V$이다.

## 모순의 논증

### 전제: QK-Norm이 커널을 바꾼다

$Q, K$에 RMSNorm을 적용하면 $\hat{q}, \hat{k} \in S^{d-1}$이 되고, 구면 항등식 $\|\hat{q} - \hat{k}\|^2 = 2 - 2\langle \hat{q}, \hat{k}\rangle$에 의해:

$$P_{ij} = \frac{e^{\beta\langle \hat{q}_i, \hat{k}_j\rangle}}{\sum_\ell e^{\beta\langle \hat{q}_i, \hat{k}_\ell\rangle}} = \frac{e^{-\frac{\beta}{2}\|\hat{q}_i - \hat{k}_j\|^2}}{\sum_\ell e^{-\frac{\beta}{2}\|\hat{q}_i - \hat{k}_\ell\|^2}}$$

$P$는 정규화된 가우시안 RBF 커널 가중치가 된다. ([[커널해석과_평균이동]] Section 1)

### 모순: $PV$는 좌표인데 변위처럼 더한다

- $P$가 확률 행렬 ($\sum_j P_{ij} = 1$)이므로 $PV = \sum_j P_{ij} V_j$는 $\{V_j\}$의 **볼록 결합** = 목적지 좌표
- 잔차 연결 $X + PV$는 "현재 좌표 + 목적지 좌표" = **좌표 공간의 기저 불일치(basis mismatch)**
- 연속 시간에서 $\dot{X} = PV$는 속도장(velocity field)에 위치(position)를 대입한 것

### 해소: $(P-I)V$

$$\Delta V = \text{Target} - \text{Current} = PV - V = (P-I)V$$

이것이 올바른 변위 벡터이며, $-(I-P)V = -L_{rw}V$ (그래프 라플라시안 확산)와 동치. ([[커널해석과_평균이동]] Section 3, 4)

## 구면 기하학에서의 추가 고려: 접선 사영

### 문제: $(P-I)V$도 구면을 벗어난다

$V$가 단위 구면 위의 점일 때, $PV$는 구면 위 점들의 볼록 결합이므로 **구의 내부**에 위치한다 (볼록 껍질의 성질).

따라서 $(P-I)V = PV - V$는 구면($V$)에서 구 내부($PV$)를 향하는 벡터로, **구의 중심을 향해 파고드는 방사형 성분(radial component)**을 포함한다.

이 벡터를 잔차에 그대로 더하면:
1. 토큰 벡터가 구면을 이탈 → 노름 감소
2. 다음 레이어의 정규화(RMSNorm)가 다시 구면으로 사영
3. 이 과정에서 **방사형 성분에 투입된 그래디언트가 완전히 소실** → 학습 낭비

### 해법: 그람-슈미트 직교화에 의한 접선 사영

$(P-I)V$를 현재 은닉 상태 $h$에 직교하는 접선 성분만 남기도록 사영:

$$O_\perp = (P-I)V W_o - \frac{\langle (P-I)V W_o,\; h \rangle}{\langle h,\; h \rangle}\, h$$

또는 단위 구면 위($\|h\| = 1$)에서:

$$O_\perp = P^\perp_h\!\bigl((P-I)V W_o\bigr) = (P-I)V W_o - \langle (P-I)V W_o,\; h\rangle\, h$$

여기서 $P^\perp_h$는 $h$의 접선 평면(tangent plane at $h$ on $S^{d-1}$)으로의 직교 사영.

### Rigollet (2024)와의 대응

이 접선 사영은 Rigollet의 SA 동역학 ([[rigollet2024]])에서 사용되는 $P^\perp_{x_i}$와 **정확히 동일한 연산**이다:

$$\dot{x}_i = P^\perp_{x_i}\!\left(\frac{\sum_j e^{\beta\langle x_i, x_j\rangle} x_j}{\sum_k e^{\beta\langle x_i, x_k\rangle}}\right)$$

수학적으로, 단위 구면 위에서:

$$P^\perp_V(PV) = PV - \langle V, PV\rangle\, V = (PV - V) - (\langle V, PV\rangle - 1)\, V$$

$$P^\perp_V((P-I)V) = (P-I)V - \langle V, (P-I)V\rangle\, V$$

$\langle V, V\rangle = 1$이므로 $P^\perp_V(V) = 0$, 따라서:

$$P^\perp_V(PV) = P^\perp_V((P-I)V)$$

**즉, "PV를 접선 사영"하는 것과 "(P-I)V를 접선 사영"하는 것은 단위 구면 위에서 동치이다.** Rigollet의 SA 동역학이 정확히 이 연산을 수행하고 있다.

### 접선 사영의 효과

| 성분 | 의미 | 접선 사영 후 |
|------|------|-------------|
| **방사형 성분** (구 중심 방향) | 노름을 줄이는 힘. 정규화가 삭제할 성분 | **제거됨** |
| **접선 성분** (구면 위 방향) | 구면 위에서의 실제 이동 방향 | **보존됨** |

접선 성분만 남기면:
- 토큰이 구면에 가깝게 유지 → 정규화의 보정 폭 최소화
- 방사형 그래디언트 소실 방지 → 학습 효율 향상
- Rigollet (2024)의 이론적 결과 (클러스터링 정리, 수렴 속도)가 직접 적용 가능

## 아키텍처 변형 후보 정리

| 변형 | 수식 | 특성 |
|------|------|------|
| **표준** | $h + PVW_o$ | QK-Norm 하에서 좌표+좌표 모순 |
| **변형 A: 단순 교체** | $h + (P-I)VW_o$ | 모순 해소, 방사형 성분 잔존 |
| **변형 B: 접선 사영** | $h + P^\perp_h((P-I)VW_o)$ | 모순 해소 + 구면 기하 정합 |

## 검토 결과 (2026-04-11)

ChatGPT 보고서 및 후속 분석에 의해 아래 사항이 교정되었다. 상세 논의는 [[PV의_기하학적_적합성]], [[잔차연결_수정안]] 참조.

1. **"수학적 모순" 표현은 과도하다.** 실제 트랜스포머의 residual stream은 $\mathbb{R}^d$의 벡터공간이며, $H + PVW_O$는 유클리드 공간에서 합법적인 연산이다. 더 정확한 표현은 **"구면 기하 해석을 채택하면 non-canonical"**이다.
2. **"basis mismatch"가 아니라 "affine update mismatch"가 정확하다.** 기저(basis)가 다른 것이 아니라, "barycentric target"을 변위(displacement)처럼 사용하는 의미론적 불일치이다.
3. **$P_V^\perp(PV) = P_V^\perp((P-I)V)$ 동치는 조건부이다.** 이 등식은 $v_iW_O \parallel h_i$일 때, 즉 값 공간의 출력이 현재 은닉 상태와 평행할 때만 성립한다. 일반적으로 $v_iW_O \neq h_i$이므로 두 사영은 다르다.
4. **$(P-I)V$의 핵심 약점**: 잔차 업데이트의 "현재점"은 $h_i$인데, $(P-I)V$가 빼는 것은 $v_i = h_iW_V$이다. 올바른 "target minus current"는 $(PVW_O)_i - h_i$이다.
5. **Identity-like head 무효화**: $P \approx I$인 head에서 $(P-I)V \approx 0$이 되어 자기 정보 통과 기능이 사라진다.
6. **Pre-RMSNorm에서 $H$는 유클리드 공간**: 접선사영 논의 자체가 불필요하며, 가장 자연스러운 수정안은 $PVW_O - H$이다.

## 미해결 사항

- [ ] 변형 A vs B의 실험적 비교
- [ ] RMSNorm 위치 (Pre-LN / Post-LN)에 따른 접선 사영의 필요성 차이
- [ ] 멀티헤드에서의 접선 사영: 각 헤드별 사영 vs 합산 후 사영
- [ ] $W_o$ 행렬과의 상호작용: 사영을 $W_o$ 전에 할지 후에 할지
- [ ] 학습 초기: $P \approx \frac{1}{n}\mathbf{11}^\top$일 때 $(P-I)V$가 거의 0 → 초기 학습 속도 문제
