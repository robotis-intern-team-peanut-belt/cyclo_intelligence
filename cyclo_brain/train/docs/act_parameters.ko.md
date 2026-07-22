# ACT 정책 파라미터 — 전체 레퍼런스

`ACTConfig`의 모든 필드, 각 필드가 무엇을 하는지, 이 저장소의 실험 YAML에서 어떻게
설정하는지, 그리고 학문적으로 어디에서 유래했는지 정리한다. 필드 목록의 기준(source of
truth)은 `cyclo_brain/policy/lerobot/lerobot/src/lerobot/policies/act/configuration_act.py`.

ACT("Action Chunking with Transformers")는 다음 논문에서 처음 소개되었다:

> Zhao, Tony Z., Kumar, Vikash, Levine, Sergey, Finn, Chelsea. **"Learning
> Fine-Grained Bimanual Manipulation with Low-Cost Hardware."** arXiv:2304.13705
> (2023). [huggingface.co/papers/2304.13705](https://huggingface.co/papers/2304.13705) ·
> [project page](https://tonyzhaozh.github.io/aloha) ·
> [original code](https://github.com/tonyzhaozh/act)

각 필드 항목에서 별도로 명시하지 않는 한, 출처는 위 논문이다. lerobot의 ACT는 여기에 더해
**DETR**에서 두 가지 구조적 요소(학습되는 디코더 쿼리, dilated-backbone 옵션)를,
**VAE** 계열 문헌에서 한 가지 요소(KL 항)를 빌려온다. 이들은 아래에서 개별적으로 인용한다.

**이 저장소에서 이 값들을 바꾸는 방법:** 실험 YAML
(`cyclo_brain/train/experiments/<name>.yaml`)의 `policy:` 아래에 있는 모든 키는
그대로 `--policy.<key>=<value>` 형태로 `lerobot-train`에 전달된다. 현재 기본값은
`_base/act.yaml`을, 자식 실험이 한두 개 키만 오버라이드하는 방식은 아무
`peanut_act_*.yaml`을 참고하라. `./docker/container.sh train-lerobot <experiment>
--dry-run`은 아무것도 실행하지 않고 최종적으로 해석된 명령을 그대로 출력한다 — 새 값을
쓸 때는 항상 이 방식으로 먼저 확인하라.

---

## 목차

1. [입력 / 출력 구조](#입력--출력-구조)
2. [비전 백본](#비전-백본)
3. [트랜스포머 아키텍처](#트랜스포머-아키텍처)
4. [VAE (CVAE) 목적함수](#vae-cvae-목적함수)
5. [추론 시점 동작](#추론-시점-동작)
6. [학습 & 손실](#학습--손실)
7. [옵티마이저 프리셋](#옵티마이저-프리셋)
8. [인접 노브 (ACT 전용 아님)](#인접-노브-act-전용-아님)
9. [권장 변경 사항](#권장-변경-사항)

---

## 입력 / 출력 구조

### `n_obs_steps`
- **기본값:** `1` (그리고 ACT에 한해서는 이것이 *유일하게* 허용되는 값)
- **영향:** 정책이 조건으로 삼을 수 있는 과거 환경 스텝(관측)의 개수.
- **변경 방법:** `policy.n_obs_steps` — 다만 `ACTConfig.__post_init__`이 `1`이
  아닌 값이면 `ValueError`를 발생시킨다. 이 필드는 공유되는 `PreTrainedConfig`
  베이스에 존재하며(Diffusion Policy나 π₀ 같은 다른 정책은 멀티 스텝 히스토리를
  사용한다), ACT의 논문/아키텍처는 항상 현재 스텝만 조건으로 삼기 때문에 이
  저장소에서는 다른 값을 실제로 쓸 일이 없다.
- **출처:** 일반 정책 설정 필드로, ACT 논문 고유의 것이 아니다.

### `chunk_size`
- **기본값:** `100` (lerobot) / **`30`** (이 저장소의 `_base/act.yaml`, 주석에
  `100, 150?`을 미검증 대안으로 표시 — [권장 변경 사항](#권장-변경-사항) 참고)
- **영향:** 액션 청킹 지평(horizon) — 트랜스포머 디코더가 한 번의 forward pass에서
  예측하는 미래 타임스텝의 개수. 이것이 바로 논문이 도입하는 *핵심* 아이디어다:
  한 번에 하나씩 액션을 예측하는 대신 청크 전체를 예측함으로써 누적 오차를 줄이고,
  정책이 다중 모드적이며 시간적으로 일관된 행동을 표현할 수 있게 한다.
- **변경 방법:** `policy.chunk_size`. 이 값은 또한
  `action_delta_indices = range(chunk_size)`를 설정하여, 학습 샘플마다 데이터셋에서
  몇 개의 미래 액션 행을 가져올지 결정한다.
- **제약:** `n_action_steps <= chunk_size` (강제됨).
- **출처:** ACT 논문, §III (action chunking).

### `n_action_steps`
- **기본값:** `100` (lerobot) / **`16`** (이 저장소)
- **영향:** 예측된 청크 중에서 실제로 로봇에 실행되는 액션의 개수. 이만큼 실행한 뒤
  정책에 새 청크를 다시 질의한다. 값이 작으면 더 자주 재계획하고(더 반응적, 추론 호출
  더 많음), `chunk_size`에 가까우면 추론 호출당 더 긴 open-loop 구간을 확정한다.
- **변경 방법:** `policy.n_action_steps`. `temporal_ensemble_coeff`가 설정된
  경우 반드시 `1`이어야 하며(아래 참고), `chunk_size`를 초과할 수 없다.
- **출처:** ACT 논문, §III.

### `input_features` / `output_features`
- **기본값:** `{}` (자동 추론)
- **영향:** 모델이 읽는 모든 것(`observation.state`, `observation.images.*`)과
  쓰는 것(`action`)에 대한 `{feature_name: PolicyFeature(type, shape)}` 딕셔너리
  그 자체. 이것이 모델이 구축되는 기준 스키마다.
- **변경 방법:** **이 저장소에서는 바꾸지 않는다.** `cyclo_train`이 실행 시점에
  데이터셋의 `meta/info.json`에서 곧바로 이 값을 유도한다(그래서 `manifest.json`이
  `action_shape`/`observation_state_shape`/`camera_keys`를 기록한다 — 자동
  추론된 내용을 기록해 두는 것이다). 이 값을 손으로 설정하는 것은 이 파이프라인
  바깥의 데이터셋에 대해 `lerobot-train`을 직접 호출할 때만 의미가 있다.
- **출처:** 일반 정책 설정 필드.

### `normalization_mapping`
- **기본값:** `{VISUAL: MEAN_STD, STATE: MEAN_STD, ACTION: MEAN_STD}`
- **영향:** 각 모달리티가 네트워크에 들어가기 전에 어떤 정규화를 적용할지 —
  `MEAN_STD`는 데이터셋 통계(`meta/stats.json`)를 이용해 z-score 정규화하고,
  `MIN_MAX`는 대신 데이터셋 min/max를 이용해 `[-1, 1]`(또는 `[0, 1]`)로 재척도화한다.
- **변경 방법:** `policy.normalization_mapping.VISUAL=...` 등. 여기의 어떤 실험
  YAML에도 노출되어 있지 않다 — 모든 런은 기본값(어디서나 MEAN_STD)을 사용하며,
  이는 ACT의 표준 방식이다.
- **출처:** 일반 정책 설정 필드; MEAN_STD 대 MIN_MAX는 표준적인 모방 학습 전처리
  선택이며, ACT 논문이 직접 규정하는 것은 아니다.

---

## 비전 백본

### `vision_backbone`
- **기본값:** `"resnet18"`
- **영향:** 각 카메라 프레임을 특징 맵으로 인코딩하는 torchvision ResNet 변형.
  `getattr(torchvision.models, vision_backbone)`로 해석되므로, torchvision이
  노출하는 어떤 ResNet 이름(`resnet18`, `resnet34`, `resnet50`, `resnet101`,
  `resnet152`)이든 동작한다 — 문자열이 `"resnet"`으로 시작해야 하도록 검증된다.
- **변경 방법:** `policy.vision_backbone=resnet34` (등). 이 저장소의 모든 런은
  기본값 `resnet18`을 사용한다 — 백본 크기에 대한 어블레이션은 아직 없다.
- **출처:** He, Kaiming, et al. **"Deep Residual Learning for Image
  Recognition."** arXiv:1512.03385 (2015). ACT 논문 자체도 모든 실제 로봇 실험에
  ResNet-18을 사용한다.

### `pretrained_backbone_weights`
- **기본값:** `"ResNet18_Weights.IMAGENET1K_V1"`
- **영향:** 백본이 ImageNet 사전학습 가중치(torchvision의 타입 지정 가중치
  문자열)에서 시작할지, 아니면 처음부터(`None`) 시작할지. 23–63k 프레임의 학습
  데이터만으로도 학습이 가능한 이유가 바로 사전학습 가중치다 — 백본은 여러분의 로봇
  데이터로부터 "엣지가 어떻게 생겼는지"를 학습하는 것이 아니라, 미세 조정만 하기
  때문이다.
- **변경 방법:** 무작위 초기화는 `policy.pretrained_backbone_weights=null`,
  또는 `vision_backbone`을 `resnet34`로 바꾼다면 예컨대
  `ResNet34_Weights.IMAGENET1K_V1`. 선택한 `vision_backbone`의 가중치 enum과
  일치해야 한다.
- **출처:** ImageNet 사전학습은 표준적인 관행이며, ACT 논문 고유의 것은 아니다(그들이
  공개한 코드도 ImageNet 초기화 ResNet-18을 기본으로 한다).

### `replace_final_stride_with_dilation`
- **기본값:** `False`
- **영향:** ResNet 마지막 스테이지의 2× 공간 다운샘플링 stride를 dilated(atrous)
  컨볼루션으로 교체한다 — 수용 영역(receptive field)은 같지만, 최종 특징 맵이 각
  공간 차원에서 **2배 높은 해상도**로 나온다. 이미지당 연산량이 더 들며, 더 미세한
  공간 정밀도가 필요한 작업(작은 물체, 빡빡한 삽입 작업)에 도움이 될 수 있다.
- **변경 방법:** `policy.replace_final_stride_with_dilation=true`. torchvision
  ResNet 생성자에 `replace_stride_with_dilation=[False, False, <this value>]`로
  그대로 전달된다(세 스테이지 중 마지막만 영향을 받는다).
- **출처:** dilated-final-stage 기법은 Carion, Nicolas, et al. **"End-to-End
  Object Detection with Transformers"** (DETR). arXiv:2005.12872 (2020)의
  "DC5" 변형이다 — ACT의 트랜스포머 헤드 설계는 DETR에서 직접 빌려왔으며, 이 옵션은
  DETR의 `--dilation` 플래그를 직접 이식한 것이다.

---

## 트랜스포머 아키텍처

이들은 비전 백본 위에 얹히는 인코더/디코더 트랜스포머를 제어한다(이미지 특징 + 로봇
상태 + 잠재 스타일 변수 `z`를 융합하고, 크로스 어텐션으로 액션 청크를 디코딩한다).
설계는 DETR의 인코더-디코더 구조를 따르며, DETR의 object query 자리에 학습되는 디코더
쿼리를 둔다 — 여기서는 객체당 하나가 아니라 청크 타임스텝당 하나의 쿼리다.

### `dim_model`
- **기본값:** `512`
- **영향:** 트랜스포머의 은닉 폭 — 모든 어텐션 블록과 잔차 스트림이 이 크기다. 클수록
  용량이 크고, 메모리를 더 쓰고, 느리다.
- **변경 방법:** `policy.dim_model`. 지금까지 모든 런에서 기본값 그대로다.
- **출처:** Vaswani, Ashish, et al. **"Attention Is All You Need."**
  arXiv:1706.03762 (2017) — 기본 트랜스포머 아키텍처. 512라는 구체적 값은 ACT
  논문 자체의 설정과 일치한다.

### `n_heads`
- **기본값:** `8`
- **영향:** 모든 멀티 헤드 어텐션 블록의 어텐션 헤드 개수(`dim_model`이 헤드 수로
  균등 분할되므로 `dim_model`을 나누어 떨어뜨려야 한다).
- **변경 방법:** `policy.n_heads`.
- **출처:** Vaswani et al. 2017 (멀티 헤드 어텐션).

### `dim_feedforward`
- **기본값:** `3200`
- **영향:** 각 트랜스포머 블록 내부 피드포워드(MLP) 서브레이어의 폭 — 흔한 "확장 후
  다시 투영" 병목. `3200 ≈ 6.25× dim_model`로, 트랜스포머 논문의 원래 4배
  경험칙보다 넓다.
- **변경 방법:** `policy.dim_feedforward`.
- **출처:** Vaswani et al. 2017; 구체적인 3200 값은 ACT 논문/코드베이스의 설정과
  일치한다.

### `feedforward_activation`
- **기본값:** `"relu"`
- **영향:** 그 피드포워드 서브레이어 내부의 비선형 함수. `"relu"`, `"gelu"`,
  `"glu"`를 허용한다(그 외는 런타임에 에러).
- **변경 방법:** `policy.feedforward_activation=gelu`.
- **출처:** ReLU는 ACT 논문의 기본값이며, GELU/GLU는 lerobot 구현에서 사용 가능한
  표준적인 트랜스포머 대안이지만 논문이나 이 저장소의 어떤 런에서도 어블레이션되지
  않았다.

### `n_encoder_layers`
- **기본값:** `4`
- **영향:** 트랜스포머 **인코더**의 깊이(크로스 어텐션 이전에 백본 이미지 토큰 + 상태
  토큰 + 잠재 `z` 토큰을 융합하는 부분). 레이어가 많을수록 디코딩 전에 서로 다른
  모달리티가 더 많이 섞인다.
- **변경 방법:** `policy.n_encoder_layers`.
- **출처:** ACT 논문 아키텍처; DETR 스타일 인코더-디코더 분할.

### `n_decoder_layers`
- **기본값:** `1`
- **영향:** 트랜스포머 **디코더**의 깊이(청크 길이의 쿼리 임베딩에서 인코더 출력으로
  크로스 어텐션하여 액션 시퀀스를 생성한다).
- **변경 방법:** `policy.n_decoder_layers`. **여기서 코드 주석이 중요하다**:
  원래 ACT 구현은 이 값을 7로 설정했지만, 그 코드의 버그로 인해 실제로는 첫 번째
  디코더 레이어만 기여했다(참고:
  [tonyzhaozh/act#25](https://github.com/tonyzhaozh/act/issues/25)). lerobot은
  이를 "고쳐서" 실질적으로 다른 모델로 만드는 대신, 원본의 실제(버그가 있는) 동작을
  *충실히 재현*하기 위해 이 값을 `1`로 고정한다. 이 값을 1보다 크게 설정하면 원래
  논문이 보고한 수치가 반영하지 않는, 실제로 더 깊은 디코더가 된다 — 버그 수정이 아니라
  미검증 영역이다.
- **출처:** ACT 논문 + 그 불일치를 문서화한 위 GitHub 이슈.

### `pre_norm`
- **기본값:** `False` (즉 post-norm, 각 서브레이어 *이후*에 LayerNorm)
- **영향:** LayerNorm을 각 어텐션/피드포워드 서브레이어 이전에(`pre_norm=True`)
  적용할지 이후에(`False`) 적용할지. Pre-norm은 일반적으로 더 높은 학습률 / 더 큰
  깊이에서 더 안정적으로 학습되며, post-norm(원래 트랜스포머의 선택이자 ACT의
  기본값)은 일단 수렴하면 약간 더 나은 성능을 낼 수 있다.
- **변경 방법:** `policy.pre_norm=true`.
- **출처:** Xiong, Ruibin, et al. **"On Layer Normalization in the
  Transformer Architecture."** arXiv:2002.04745 (2020) — pre-LN 대 post-LN을
  형식화하고 비교한 논문.

---

## VAE (CVAE) 목적함수

ACT는 조건부 VAE를 학습한다: 학습 중에 작은 트랜스포머 인코더("VAE 인코더")가 정답
액션 청크 + 로봇 상태를 보고 잠재 스타일 변수 `z`를 생성하고, 그다음 메인 인코더-디코더가
`z`에 *조건화되어* 액션 청크를 예측한다. 추론 시점에는 인코딩할 정답 청크가 없으므로
`z`는 그냥 0으로 설정된다.

### `use_vae`
- **기본값:** `True`
- **영향:** CVAE 기제(인코더 + KL 손실)를 사용할지 여부. `False`이면 ACT가 순수한
  결정론적 청크 액션 예측기로 붕괴한다(`z` 없음, KL 항 없음).
- **변경 방법:** `policy.use_vae=false`. 지금까지 모든 런은 이를 켜 둔다(기본값이며,
  논문이 주요 수치를 보고할 때 쓴 설정).
- **출처:** ACT 논문 §III-B, "Modeling human data with a conditional VAE";
  조건부 VAE 형식 자체는 Sohn, Kihyuk, Lee, Honglak, and Yan, Xinchen.
  **"Learning Structured Output Representation using Deep Conditional
  Generative Models."** NeurIPS 2015에서 유래.

### `latent_dim`
- **기본값:** `32`
- **영향:** 잠재 스타일 변수 `z`의 차원. VAE 기준으로는 일부러 작게 잡았다 — `z`는
  큰 생성 코드가 아니라, 시연자가 작업을 수행한 *스타일적* 변이(속도, 정확한 궤적)를
  포착하기 위한 것이다.
- **변경 방법:** `policy.latent_dim`.
- **출처:** ACT 논문 §III-B; VAE 잠재 공간 개념은 Kingma & Welling(아래)에서 유래.

### `n_vae_encoder_layers`
- **기본값:** `4`
- **영향:** VAE 인코더 트랜스포머의 깊이(`[cls, robot_state, *action_sequence]`를
  잠재 분포의 평균/로그 분산까지 처리한다). `use_vae=True`일 때만 의미가 있다.
- **변경 방법:** `policy.n_vae_encoder_layers`.
- **출처:** ACT 논문 아키텍처.

---

## 추론 시점 동작

### `temporal_ensemble_coeff`
- **기본값:** `None` (비활성화)
- **영향:** `n_action_steps` 청크 전체를 open-loop로 실행하는 것의 대안. 이 값이
  설정되면 정책이 **매** 스텝마다 질의되고(`n_action_steps=1`을 강제), 실제로
  실행되는 액션은 최근 여러 질의에서 나온, 겹치는 청크 예측들에 대한 지수 가중 평균이
  된다 — 더 최신의 예측일수록 `exp(-coeff * age)`로 가중된다. 청크 경계 사이의
  지터(jitter)를 매끄럽게 하지만, `n_action_steps`당 한 번이 아니라 환경 스텝당 한
  번의 추론 호출을 대가로 한다.
- **변경 방법:** `policy.temporal_ensemble_coeff=0.01` — 이 기능을 사용할 때
  논문 자체의 기본값. `ACTTemporalEnsembler`로 구현되어 있다.
- **출처:** ACT 논문, Algorithm 2 ("temporal ensembling"). 이 저장소의 어떤
  런에서도 사용되지 않는다 — 모든 런은 고정 청크 open-loop 모드
  (`n_action_steps=16`)를 대신 사용한다.

---

## 학습 & 손실

### `dropout`
- **기본값:** `0.1`
- **영향:** 학습 중 트랜스포머 서브레이어 전반에 적용되는 드롭아웃 확률(과적합에 대한
  정규화). 여기의 모든 런에서 기본값 그대로다.
- **변경 방법:** `policy.dropout`.
- **출처:** Srivastava, Nitish, et al. **"Dropout: A Simple Way to Prevent
  Neural Networks from Overfitting."** JMLR 15 (2014). ACT 논문 고유가 아닌
  일반적인 정규화 기법이며, ACT의 코드도 관례적인 0.1 값을 쓸 뿐이다.

### `kl_weight`
- **기본값:** `10.0`
- **영향:** VAE 손실에서 KL 발산 항에 붙는 가중치:
  `loss = reconstruction_loss + kl_weight * kl_divergence(latent_pdf ||
  standard_normal)`. 값이 높을수록 `z`를 표준 정규분포 쪽으로 더 강하게 밀어붙이고(더
  정규화되고, 덜 유용한 잠재, 더 결정론적으로 느껴지는 정책), 값이 낮을수록 `z`가
  시연 스타일에 대한 정보를 더 많이 담게 한다(순수 재구성에 가까워지지만, 추론 시점에는
  `z=0`이므로 그때 사용할 수 없는 행동을 잠재가 "누출"할 수 있다).
- **변경 방법:** `policy.kl_weight`. **이 저장소는 이 값을 어블레이션했다**:
  F2 stage-2와 SG2 stage-3 데이터셋 양쪽에서 `kl_weight=10`(기본값) 대
  `kl_weight=5` — 왜 결과가 결론에 이르지 못했는지는 [권장 변경
  사항](#권장-변경-사항) 참고.
- **출처:** KL 항 자체는 Kingma, Diederik P., and Welling, Max.
  **"Auto-Encoding Variational Bayes."** arXiv:1312.6114 (2013)의 표준 VAE
  손실이다(lerobot의 KL 계산이 이를 직접 인용, App. B). 구체적인 `kl_weight`
  *하이퍼파라미터* — 즉 그 KL 항을 가중치 1로 쓰는 대신 베타 가중하는 방식 — 은
  Higgins, Irina, et al. **"β-VAE: Learning Basic Visual Concepts with a
  Constrained Variational Framework."** ICLR 2017의 beta-VAE 틀을 따른다. ACT
  논문 자체(Table III)는 모든 실험에 걸쳐 단일 고정값 **β = 10**을 균일하게 썼다고
  보고하며, 작업별로 값을 스윕하거나 다른 값이 필요했다고 보고하지 않는다. 이 저장소의
  기본값 `10.0`은 이와 직접 일치하며, 여기서 테스트한 `5.0` 변형은 논문 자체가
  동기를 부여한 것이 아니라 이 프로젝트 고유의 어블레이션이다.

---

## 옵티마이저 프리셋

ACT는 상위 학습 설정의 옵티마이저 블록에 의존하는 대신 이 세 가지의 자체 사본을
정의한다(`get_optimizer_preset()`이 이들로부터 `AdamWConfig`를 구축한다) — 그래서
이들은 YAML에서 `train:`이 아니라 `policy:` 아래에 있다.

### `optimizer_lr`
- **기본값:** `1e-5`
- **영향:** 비전 백본을 *제외한* 모든 파라미터의 학습률.
- **변경 방법:** `policy.optimizer_lr`.
- **출처:** AdamW 옵티마이저 자체 — Loshchilov, Ilya, and Hutter, Frank.
  **"Decoupled Weight Decay Regularization."** arXiv:1711.05101 (2019). `1e-5`
  값은 ACT 논문의 설정과 일치한다.

### `optimizer_lr_backbone`
- **기본값:** `1e-5`
- **영향:** ResNet 백본의 파라미터에 특화된 **별도의** 학습률
  (`get_optim_params()`이 백본과 나머지 전부를 두 개의 옵티마이저 파라미터 그룹으로
  분리한다). 그 발상(검출/DETR 스타일 전이 학습에서 표준)은, 사전학습된 CNN 백본이
  보통 그 위에서 처음부터 학습되는 트랜스포머 헤드보다 작은 학습률을 원한다는 것이다.
- **변경 방법:** `policy.optimizer_lr_backbone`. **현재는 실질적으로 죽은
  노브**: 코드에 `TODO(aliberts, rcadene): As of now, lr_backbone == lr`라는
  주석이 있고, 실제로 둘 다 `1e-5`로 기본 설정되어 있다 — 이 저장소(또는 lerobot의
  기본값)의 어떤 런도 아직 둘을 실제로 구분하지 않는다.
- **출처:** 백본/헤드 차등 학습률은 DETR의 관례(Carion et al. 2020); AdamW는
  위와 같다.

### `optimizer_weight_decay`
- **기본값:** `1e-4`
- **영향:** 비백본 파라미터 그룹에 대한 L2 스타일 weight decay(AdamW의 decoupled
  변형).
- **변경 방법:** `policy.optimizer_weight_decay`.
- **출처:** Loshchilov & Hutter 2019 (decoupled weight decay).

---

## 인접 노브 (ACT 전용 아님)

이들은 `ACTConfig`가 아니라 일반 `TrainPipelineConfig`(YAML의 `train:`)에 있으며 —
lerobot의 모든 정책이 공유한다. 이 저장소의 모든 런이 이들을 건드리고 ACT가 학습되는
방식을 실질적으로 바꾸기 때문에 여기 실었지만, 위의 "ACT 파라미터" 목록에 속하지는
않는다.

| 키 | 기본값 | 역할 |
|---|---|---|
| `train.batch_size` | 8 | 그래디언트 스텝당 샘플 수. |
| `train.steps` | 100,000 | 총 옵티마이저 스텝. `batch_size` 및 데이터셋 프레임 수와 결합되어 **에폭**을 결정한다 — [권장 변경 사항](#권장-변경-사항) 참고. |
| `train.save_freq` | 10,000 (이 저장소) | `output/checkpoints/` 아래에 체크포인트를 얼마나 자주 쓸지. |
| `train.log_freq` | 200 | 손실을 stdout/wandb에 얼마나 자주 로깅할지. |
| `train.num_workers` | 4 | 데이터로더 워커 프로세스(병목은 CPU 바운드 Python이 아니라 비디오 디코드다). |
| `train.seed` | 1000 | 전역 RNG 시드. |
| `policy.device` | `cuda` | `PreTrainedConfig`에도 있음 — `cuda`, `cuda:N`, `cpu`, `mps`. 이 장비(단일 RTX 5090)에서는 `cuda:N`이 무의미하다; `accelerate`가 디바이스를 고른다. |
| `policy.use_amp` | `False` | `PreTrainedConfig`에도 있음 — 혼합 정밀도 학습을 활성화한다. 아직 어떤 런에서도 시도되지 않았다; 특히 더 느린 4카메라 SG2 런에서 처리량 이득이 있을 법하다. |

---

## 권장 변경 사항

지금까지 이 저장소에서 실제로 수행한 런(완료된 baseline/KL 런 5개 + 6런 chunk-size
어블레이션)에 근거한 것으로, 단순한 논문 기본값이 아니다.

1. **`chunk_size` 설정을 원(raw) 학습 손실로 비교하지 마라.** 방금 완료된
   어블레이션은 두 데이터셋 모두에서 손실이 `chunk_size`에 따라 단조적으로 *증가*함을
   보여준다(SG2 stage-3: chunk 30/60/100/150에 대해 0.026 → 0.029 → 0.034 →
   0.037; F2 stage-2도 비슷하게 증가). 이는 예상된 것이며 회귀 신호가 아니다 — 더
   먼 미래를 예측하는 것은 엄밀히 더 어려운 재구성 목표이므로, `kl_weight` 스윕을
   고정 `chunk_size`에서 비교할 때와 달리 손실 스케일이 chunk 크기 간에 서로 비교
   가능하지 않다. **여기서 실제로 승자를 고르는 유일한 방법은 하드웨어에서의 closed-loop
   롤아웃 성공률**이지, `train.log` 수치가 아니다.

2. **`kl_weight=10` 대 `kl_weight=5`는 테스트 설계상 결론에 이르지 못했다.** 두
   로봇 모두에서 최종 손실이 서로 0.001 이내로 들어왔다. 이는 `kl_weight`가 잠재
   정규화 대 재구성 충실도를 절충한다는 사실과 일관된다 — 이 절충은 학습 손실이 아니라
   *closed-loop 행동*에서 드러난다(정책이 시연의 한 모드에 확신을 가지고 커밋하는가,
   아니면 스타일들을 회피/평균하는가?). `kl_weight` 값을 더 재실행해도 이는 해소되지
   않는다; 기존 두 체크포인트 간의 on-robot A/B가 해소할 것이다.

3. **F2 stage-2는 어떤 하이퍼파라미터 선택과도 무관하게 과적합 위험이 있다.** 100k
   스텝에서 53 에피소드 / 23.4k 프레임 = 34 에폭으로, 더 큰 SG2 데이터셋(221–297
   에피소드)이 자체 80–100k 스텝 런에서 도달하는 ~12–14 에폭 범위의 약 2.5배다.
   이것이 바로 이 저장소에서 F2 chunk-size 어블레이션이 100k를 반복하는 대신
   의도적으로 40k 스텝(~13.7 에폭)을 쓰는 *이유*다 — 앞으로의 F2 실험이 그 과적합
   위험을 재수용하겠다고 의도적으로 결정하지 않은 채 100k로 되돌아가게 두지 마라. F2에
   대한 가장 레버리지가 큰 다음 단계는 하이퍼파라미터 탐색이 아니라 **더 많은 에피소드**다.

4. **`use_amp`는 한 번도 시도되지 않았다.** SG2 런이 느린 쪽이다(4카메라, ~4.2
   step/s 대 F2의 ~9.5 step/s) — 혼합 정밀도는 특히 거기서 공짜 처리량 이득일 법하며,
   전체 런에서 신뢰하기 전에 KL 손실 항을 불안정하게 만들지 않는지 확인하는 짧은 스모크
   테스트(먼저 `--dry-run`, 그다음 짧은 `--set train.steps=2000` 런)를 해 볼 가치가
   있다.

5. **`vision_backbone`은 한 번도 어블레이션되지 않았다.** 모든 런이 `resnet18`을
   쓰며, 이는 ACT 논문 자체의 선택이기도 하고 이 작업들의 병목일 가능성이 낮다 — 특정
   실패 모드(예: 작은 물체 놓침)가 더 큰 백본 용량이 도움이 될 것임을 시사하지 않는
   한 우선순위를 낮춰라.

6. **`n_decoder_layers`는 아마 `1`로 유지하는 게 좋다.** 이는 1개 레이어가
   아키텍처적으로 권장되어서가 아니라 원래 ACT 논문의 실제(버그가 있지만 보고된) 동작에
   맞추기 위해 거기에 고정된 것이다. 그 너머를 탐색한다면 버그 수정이 아니라 "논문과
   실질적으로 다른 모델"로 취급하라 — 가벼운 상향이 아니라, 자체 eval을 갖춘 독립적인
   어블레이션이 필요하다.

7. **`optimizer_lr_backbone`은 현재 `optimizer_lr`에 대해 사실상 no-op이다**(둘
   다 `1e-5`). 앞으로 어떤 런이 사전학습된 ImageNet 백본이 처음부터 학습되는
   트랜스포머 헤드만큼 빠르게 표류하지 않도록 보호하고 싶다면(작은 데이터셋을 미세
   조정할 때의 표준 관행 — 특히 F2의 53 에피소드에 관련), 이것이 분리해야 할
   노브다. 예: `optimizer_lr=1e-5`는 그대로 두고 `optimizer_lr_backbone=1e-6`.
   여기서는 미검증.
