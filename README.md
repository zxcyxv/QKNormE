# QK-Norm Gaussian: 트랜스포머 잔차 연결의 기하학 연구

QK-Norm Gaussian은 QK-Norm이 트랜스포머 attention의 기하학적 해석을 어떻게 바꾸는지, 그리고 그 해석이 잔차 연결(residual connection)의 다른 설계를 제안하는지 탐구하는 연구용 코드베이스입니다.

이 프로젝트는 세 부분으로 구성됩니다.

- QK-Norm Transformer 변형을 비교하는 PyTorch 실험 코드
- 논문 요약, 개념 정리, 보고서를 누적하는 Markdown 연구 위키
- 로컬 위키와 원본 PDF를 활용해 연구 보고서를 생성하는 OpenAI Responses API 스크립트

중심 질문은 다음입니다.

> QK-Norm으로 attention weight를 Gaussian/von Mises-Fisher kernel weight처럼 해석할 수 있다면, attention 출력은 잔차에 더할 변위(displacement)가 아니라 barycentric target으로 보아야 하는가?

최종 결론은 보수적입니다. QK-Norm의 kernel 해석은 유용하지만, 그것만으로 표준 residual attention이 수학적으로 틀렸다고 결론낼 수는 없습니다. 더 강한 주장을 하려면 일반 Transformer가 만족하지 않는 추가 전제가 필요합니다.

## 저장소 구조

```text
experiment/
  config.py        실험 하이퍼파라미터와 variant 생성 함수
  data.py          Wikitext-103 토크나이징, chunking, 캐싱
  metrics.py       attention entropy, gradient norm, tau/eta, alignment 지표
  model.py         QK-Norm GPT 모델과 잔차 연결 변형
  run_all.py       모든 variant 순차 실행 스크립트
  train.py         학습 루프, 평가, 로그, 체크포인트 저장

raw/
  내 제안.txt       초기 연구 제안 메모
  *.pdf            수정하지 않는 원본 논문 파일

wiki/
  index.md         위키 목차
  concepts/        개념별 종합 정리
  sources/         논문 및 원본 자료 요약
  reports/         생성/정리된 연구 보고서

report.py          OpenAI Responses API 기반 보고서 생성기
prompt_draft.md    외부 보고서 생성을 위한 초기 프롬프트 초안
CLAUDE.md          위키 스키마와 관리 규칙
llm-wiki.md        지속적 연구 위키 워크플로우 설명
```

## 구현된 실험 변형

모델은 RMSNorm, causal self-attention, QK-Norm, feed-forward block, final RMSNorm, tied token/output embedding을 사용하는 작은 GPT-style decoder입니다.

Attention 모듈은 항상 일반적인 projected attention 출력을 계산합니다.

```text
Attn(H) = P V W_O
```

잔차 연결 방식은 [experiment/config.py](/workspace/QKNormE/experiment/config.py)의 `attn_output_mode`로 선택합니다.

### 1. Baseline

표준 Pre-RMSNorm 잔차 업데이트입니다.

```text
H <- H + Attn(RMSNorm(H))
H <- H + FFN(RMSNorm(H))
```

### 2. ReZero

잔차 branch를 0으로 초기화된 learnable scalar로 조절합니다.

```text
H <- H + eta_attn * Attn(H)
H <- H + eta_ffn  * FFN(H)
```

초기에는 identity mapping으로 시작하고, 학습이 진행되며 잔차 branch의 영향이 점진적으로 커집니다.

### 3. ReZero + PVH

Attention 출력을 target으로 보고 현재 hidden state와 보간하는 변형입니다.

```text
H <- (1 - eta_attn) * H + eta_attn * Attn(H)
```

동치로 쓰면 다음과 같습니다.

```text
H <- H + eta_attn * (P V W_O - H)
```

이 변형은 attention output이 변위가 아니라 target에 가깝다는 가설을 실험하기 위한 최소한의 probe입니다. 엄밀한 Mean-Shift update라고 주장하지 않습니다.

## 설치

Python 3.11 이상을 사용합니다.

```bash
pip install uv
uv pip install -e .
```

현재 `pyproject.toml`은 `torch`를 CUDA 12.8 wheel index에서 받도록 설정되어 있습니다. CPU-only 환경에서는 CPU용 Torch를 별도로 설치하거나, Torch source 설정을 조정한 뒤 전체 의존성을 설치하는 편이 좋습니다.

학습 스택 전체를 설치하지 않고 PDF나 위키만 가볍게 확인하려면 다음처럼 프로젝트 의존성 동기화를 건너뛸 수 있습니다.

```bash
uv run --no-project --with pymupdf python -c "import fitz; print(fitz.__doc__)"
```

## 학습 실행

단일 variant 학습:

```bash
uv run python -m experiment.train --variant baseline
uv run python -m experiment.train --variant rezero
uv run python -m experiment.train --variant rezero_pvh
```

모든 variant 순차 실행:

```bash
uv run python -m experiment.run_all
```

주요 옵션 예시:

```bash
uv run python -m experiment.train \
  --variant rezero_pvh \
  --max-steps 20000 \
  --batch-size 16 \
  --lr 3e-4 \
  --seed 42
```

학습 결과는 `runs/<variant>_seed<seed>/`에 저장되며 git에는 포함하지 않습니다.

## 연구 위키

`wiki/`는 단순 검색 인덱스가 아니라 지속적으로 축적되는 연구 노트입니다. 원본 논문은 `wiki/sources/`에 요약하고, 개념적 연결은 `wiki/concepts/`에 통합하며, 긴 분석은 `wiki/reports/`에 저장합니다.

주요 문서:

- [wiki/index.md](/workspace/QKNormE/wiki/index.md): 위키 목차
- [wiki/concepts/커널해석과_평균이동.md](/workspace/QKNormE/wiki/concepts/커널해석과_평균이동.md): QK-Norm의 kernel 해석과 Mean-Shift 논리
- [wiki/concepts/PV의_기하학적_적합성.md](/workspace/QKNormE/wiki/concepts/PV의_기하학적_적합성.md): `PV`를 잔차 출력으로 쓰는 것의 기하학적 적합성 검토
- [wiki/concepts/잔차연결_수정안.md](/workspace/QKNormE/wiki/concepts/잔차연결_수정안.md): `(P-I)V`, `PVW_O-H`, tangent projection, nGPT식 재설계 비교
- [wiki/reports/최종결론.md](/workspace/QKNormE/wiki/reports/최종결론.md): 초기 제안에서 생략된 전제를 비판적으로 정리한 최종 보고서

## 보고서 생성

[report.py](/workspace/QKNormE/report.py)는 OpenAI Responses API로 연구 보고서를 생성합니다. 로컬 위키 내용을 context로 포함할 수 있고, PDF나 원본 파일을 vector store에 업로드해 file search를 사용할 수 있습니다.

환경변수:

```bash
export OPENAI_API_KEY=...
```

실행 예시:

```bash
uv run python report.py \
  --wiki \
  --prompt prompt_draft.md \
  --sources raw/2512.01868v4.pdf raw/2305.05465v6.pdf
```

생성된 보고서는 `wiki/reports/`에 저장됩니다.

## 연구 결론

초기 제안은 다음 관찰에서 출발했습니다.

1. QK-Norm은 query와 key vector의 norm을 고정한다.
2. 고정 반지름 구면 위에서는 inner product attention을 Gaussian RBF 또는 vMF kernel로 다시 쓸 수 있다.
3. 따라서 attention matrix `P`는 normalized kernel weight matrix로 해석될 수 있다.
4. Weighted average `PV`는 변위가 아니라 barycentric target처럼 보인다.

이 관찰은 표준 residual output을 `(P-I)V` 같은 차분항으로 바꾸자는 제안으로 이어졌습니다.

최종 분석은 이 직접 결론을 기각합니다. 빠진 전제는 Mean-Shift가 같은 변수를 kernel 생성, 평균 대상, 현재점으로 동시에 사용한다는 점입니다.

```text
Mean-Shift: target - current = P(X)X - X
```

실제 Transformer에서는 이 역할들이 분리되어 있습니다.

```text
q_i = h_i W_Q
k_j = h_j W_K
v_j = h_j W_V
target_i = sum_j P_ij v_j W_O
current_i = h_i
```

즉 Gaussian kernel 해석은 Q/K 공간에서 생기고, 평균은 V/O 공간의 feature에 대해 이루어지며, residual update는 H 공간에서 일어납니다. 이 때문에 엄밀한 Mean-Shift 동일시는 깨집니다.

수정된 결론은 다음과 같습니다.

- `(P-I)V`는 일반 Transformer에서 올바른 residual displacement가 아닙니다. 실제 residual state인 `h_i`가 아니라 `v_i`를 빼기 때문입니다.
- `PVW_O - H`는 hidden-space target-current probe로는 더 낫지만, kernel space, value space, output space, hidden state space가 동일시되지 않는 한 엄밀한 Mean-Shift update는 아닙니다.
- 문자 그대로의 Mean-Shift 등가를 원하면 `Q=K=V=I`, `W_O=I`, hidden state가 구면 위에 있다는 강한 전제가 필요합니다.
- 구면 해석을 진지하게 아키텍처에 반영하려면 `(P-I)V`를 국소적으로 끼워 넣는 것보다 nGPT식 full normalization이 더 정합적입니다. 즉 hidden state, block output, embedding, 관련 matrix vector를 같은 구면 구조에 놓고 normalized target-current interpolation을 수행해야 합니다.

따라서 이 프로젝트의 질문은 다음처럼 수정됩니다.

> QK-Norm의 kernel 해석이 단순 분석 도구를 넘어 아키텍처 원리가 되려면, Transformer의 어느 부분까지 공통 기하학적 공간에 강제해야 하는가?

현재 구현된 `rezero_pvh` variant는 이 방향을 탐색하는 최소 실험이지, 이론적으로 완성된 최종 아키텍처가 아닙니다.
