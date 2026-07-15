# cyclo-train

Cyclo Intelligence의 설정 기반(config-driven) 정책 학습 도구입니다. **학습 1회 = YAML 1개.**

모든 작업은 하나의 명령으로 실행합니다:

```bash
./docker/container.sh train-lerobot <experiment>
```

English: **[README.md](README.md)**

---

## 목차

1. [빠른 시작](#빠른-시작)
2. [전체 워크플로우 예시](#전체-워크플로우-예시)
3. [명령어](#명령어)
4. [실험(experiment) 작성하기](#실험experiment-작성하기)
5. [파일이 저장되는 위치](#파일이-저장되는-위치)
6. [Weights & Biases](#weights--biases)
7. [GPU 선택](#gpu-선택)
8. [각 파일의 역할](#각-파일의-역할)
9. [개발](#개발)

---

## 빠른 시작

```bash
# 실제 실행 없이 학습 명령어만 확인
./docker/container.sh train-lerobot peanut_act_80k --dry-run

# 학습 시작 (tmux 세션에서 백그라운드 실행)
./docker/container.sh train-lerobot peanut_act_80k

# 진행 상황 보기
tmux attach -t cyclo_train

# 학습 목록 확인
./docker/container.sh train-lerobot --list

# 완료된 학습을 추론용으로 배포
./docker/container.sh train-lerobot --promote peanut_act_80k
```

작은 데이터셋으로 30초 만에 전체 동작을 확인하려면:

```bash
./docker/container.sh train-lerobot act_smoke --no-tmux --set tracking.backend=none
```

---

## 전체 워크플로우 예시

새로 녹화한 태스크로 ACT 정책을 학습하는 전 과정입니다.

### 1. 데이터셋 받기

데이터셋은 HuggingFace Hub에서 받습니다. `workspace/dataset/<name>`에 다운로드되고,
v2.1이면 자동으로 v3.0으로 변환됩니다.

```bash
./docker/container.sh train-lerobot download RobotisSW/Task_900010_MyTask_lerobot
```

최종 경로와 v2.1 백업 여부가 출력됩니다:

```
[cyclo-train] dataset: /workspace/dataset/Task_900010_MyTask_lerobot  (v3.0)
[cyclo-train] v2.1 backup kept at: /workspace/dataset/Task_900010_MyTask_lerobot_old
```

### 2. 모바일 베이스 차원 제거 (해당하는 경우)

FFW-SG2는 22개 DOF를 기록합니다: 본체 관절 19개 + 모바일 베이스 3개
(`linear_x`, `linear_y`, `angular_z`). 고정된 위치에서 수행하는 태스크라면 19차원으로 학습합니다:

```bash
./docker/container.sh train-lerobot trim-mobile Task_900010_MyTask_lerobot
```

```
[cyclo-train] trimmed 22 -> 19 dims: /workspace/dataset/Task_900010_MyTask_lerobot_no_mobile
```

원본은 그대로 유지되며, 복사본은 하드링크를 사용하므로 영상 파일에 추가 디스크가 들지 않습니다.

> 학습에 사용한 차원 수는 추론 시 엔진이 구성하는 관측값과 반드시 일치해야 합니다
> (`lerobot_engine`의 `_state_modalities`). 19차원으로 학습했다면 엔진에서 `mobile`
> 모달리티를 포함하면 안 됩니다.

### 3. 실험 파일 작성

`experiments/task_900010_myTask_act.yaml`:

```yaml
_base: _base/act.yaml

name: Task_900010_MyTask_act_80k

dataset:
  repo_id: RobotisSW/Task_900010_MyTask_lerobot
  root: ${workspace}/dataset/Task_900010_MyTask_lerobot_no_mobile
```

### 4. GPU를 하루 쓰기 전에 확인

```bash
./docker/container.sh train-lerobot task_900010_myTask_act --dry-run
```

`_base`를 병합하고, `${workspace}`를 치환하고, 데이터셋 존재 여부를 검증한 뒤
실제 `lerobot-train` 명령어를 출력합니다. 아무것도 기록하지 않습니다.

### 5. 학습

```bash
./docker/container.sh train-lerobot task_900010_myTask_act
```

```
[cyclo-train] Starting run 'Task_900010_MyTask_act_80k' (lerobot)
[cyclo-train] run dir:  .../docker/workspace/runs/Task_900010_MyTask_act_80k
[cyclo-train] log:      .../runs/Task_900010_MyTask_act_80k/train.log
[cyclo-train] tracking: wandb → cyclo-lerobot
[container.sh] Training started in tmux session: cyclo_train
```

진행 상황 확인:

```bash
tmux attach -t cyclo_train                       # Ctrl-b d 로 빠져나오기
tail -f docker/workspace/runs/Task_900010_MyTask_act_80k/train.log
```

### 6. 중단된 경우 이어서 학습

```bash
./docker/container.sh train-lerobot task_900010_myTask_act --resume
```

마지막 체크포인트에서 원래 지정한 step 수까지 이어서 학습합니다. 이전 시도의
로그와 manifest는 그대로 보존됩니다.

### 7. 수렴 시점 판단

```bash
./docker/container.sh train-lerobot analyze-loss Task_900010_MyTask_act_80k --interval 10000
```

```
step       loss       abs_change   rel_change   stable
    10,000   0.512340     -            -            no
    20,000   0.221010   0.291330       56.86%       no
    ...
Recommended step: 60,000
```

### 8. 추론용으로 배포

```bash
./docker/container.sh train-lerobot --list
./docker/container.sh train-lerobot --promote Task_900010_MyTask_act_80k
# 특정 체크포인트를 지정하려면:
./docker/container.sh train-lerobot --promote Task_900010_MyTask_act_80k --step 060000
```

모델이 `workspace/model/lerobot/<name>/`에 생성되어 Inference 페이지에서 선택할 수
있게 됩니다. 공유가 필요하면 UI(Edit Dataset → Hugging Face)에서 Hub로 업로드하세요.

---

## 명령어

모두 `./docker/container.sh train-lerobot …` 형태입니다.

| 명령어 | 설명 |
|---|---|
| `<experiment>` | 학습 시작 |
| `<experiment> --resume` | 마지막 체크포인트에서 이어서 학습 |
| `<experiment> --dry-run` | 실행 없이 최종 학습 명령어만 출력 |
| `<experiment> --set K=V` | 설정 값 덮어쓰기 (여러 번 사용 가능) |
| `--list` | 학습 목록, 체크포인트, 배포 여부 확인 |
| `--promote <run>` | 체크포인트를 추론용 위치로 복사 |
| `download <repo_id>` | Hub에서 데이터셋 다운로드 + v2.1 → v3.0 변환 |
| `convert <dataset>` | 로컬 v2.1 데이터셋을 v3.0으로 변환 (`--no-mobile`로 차원 제거 동시 수행) |
| `trim-mobile <dataset>` | v3.0 데이터셋에서 모바일 3개 DOF 제거 |
| `analyze-loss <run>` | 학습 로그에서 loss 수렴 구간 탐지 |
| `--help` | 전체 옵션 |

유용한 플래그: `--no-tmux`(포그라운드 실행), `--session NAME`(동시 학습),
`--build`(학습 이미지 재빌드).

---

## 실험(experiment) 작성하기

고정된 기본값을 상속하고, 바뀌는 값만 덮어씁니다:

```yaml
_base: _base/act.yaml

name: my_run                      # 학습 디렉터리와 배포 모델 이름. 고유해야 합니다.

dataset:
  repo_id: RobotisSW/Task_900006_..._v30    # 식별자 / 이력 기록용
  root: ${workspace}/dataset/Task_900006_..._no_mobile   # 실제로 읽는 경로

policy:
  kl_weight: 5.0                  # 기본값과의 유일한 차이
```

- `${workspace}`와 `${name}`은 자동으로 치환됩니다.
- `policy:` 아래 키는 `--policy.<key>=<value>`로, `train:` 아래 키는
  `--<key>=<value>`로 전달됩니다. 따라서 모든 lerobot 플래그를 Python 수정 없이
  YAML에서 사용할 수 있습니다.
- 변형을 만들 때는 실험 파일을 복사하지 말고 `_base`를 사용해 새 설정을 추가하세요.

파일을 수정하지 않고 일회성으로 값을 바꾸려면:

```bash
./docker/container.sh train-lerobot peanut_act_80k --set train.steps=2000 --set name=quick_probe
```

지수 표기는 YAML 형식으로 씁니다: `--set policy.optimizer_lr=1.0e-5`

---

## 파일이 저장되는 위치

```
workspace/                                  # git에서 제외된 데이터 영역
├── dataset/<name>/                         # 데이터셋
├── runs/<name>/                            # 학습 1회당 디렉터리 1개
│   ├── manifest.json                       #   어떤 코드와 데이터로 만들어졌는지
│   ├── resolved_config.yaml                #   최종 병합된 설정
│   ├── train.log                           #   학습 전체 출력
│   └── output/checkpoints/{000010,…,last}/ #   체크포인트
└── model/lerobot/<name>/                   # 추론용 위치 — `promote`만 여기에 씁니다
    ├── checkpoints/<step>/pretrained_model/
    └── manifest.json
```

학습은 `runs/`에만 기록합니다. `model/`에 쓰는 것은 `promote`뿐이므로, 직접 선택한
체크포인트만 추론에서 사용할 수 있습니다. `promote`는 `pretrained_model/`만 복사하며,
옵티마이저 상태(체크포인트당 수백 MB, 이어서 학습할 때만 필요)는 배포/업로드 대상에서
제외됩니다.

`workspace/`는 컨테이너가 root 권한으로 기록하므로, 학습 결과를 지우려면
`sudo rm -rf docker/workspace/runs/<name>`이 필요합니다.

### manifest

모든 학습은 시작 전에 자신의 생성 정보를 기록합니다:

| 필드 | 내용 |
|---|---|
| `code.repo_git_sha`, `repo_dirty` | 어떤 코드인지 |
| `code.lerobot_submodule_sha` | 어떤 학습기인지 |
| `code.image` | 어떤 이미지인지 |
| `config_hash`, `config` | 어떤 설정인지 |
| `dataset.action_shape`, `observation_state_shape` | 19차원 / 22차원 |
| `runtime.gpus`, `torch`, `cuda_visible_devices` | 어떤 하드웨어인지 |
| `argv` | 실제 실행된 명령어 |
| `resumes[]` | 이어서 학습한 기록 |

`promote` 시 모델과 함께 복사되므로, 배포된 모델은 항상 자신의 이력을 갖습니다.

---

## Weights & Biases

```bash
cp docker/.env.example docker/.env      # git에서 제외됨
# WANDB_API_KEY=<https://wandb.ai/authorize 에서 발급> 추가
```

실험별로 설정합니다 (전체 적용은 `_base/act.yaml`에서):

```yaml
tracking:
  backend: wandb          # wandb | none
  project: cyclo-lerobot
  entity: robotis-intern-team-peanut-belt   # 계정이 속한 entity여야 합니다
  mode: online            # online | offline | disabled
```

학습 시작 시 run URL이 출력됩니다. 지표는 `train.log_freq` step마다 기록되며
(기본 200), 그보다 짧은 학습은 그래프가 남지 않습니다.

`wandb`를 사용할 수 없으면 경고를 출력하고 로컬 로그만으로 학습을 계속합니다.
오류로 처리하려면 `--strict-tracking`을 사용하세요. 체크포인트는 이미 디스크에
있으므로 wandb 아티팩트로 업로드하지 않습니다.

---

## GPU 선택

```bash
CUDA_VISIBLE_DEVICES=1 ./docker/container.sh train-lerobot peanut_act_80k
```

`policy.device: cuda:1`이 아니라 `CUDA_VISIBLE_DEVICES`를 사용하세요. 학습 경로는
HuggingFace `accelerate`를 사용하는데, 장치를 자체적으로 선택하며 `policy.device`는
CPU 강제 지정에만 참조합니다. 모든 GPU를 쓰려면 이 변수를 **설정하지 않은 상태로**
두세요. 빈 값으로 두면 모든 GPU가 보이지 않게 됩니다.

---

## 각 파일의 역할

### 설정

| 파일 | 역할 |
|---|---|
| `experiments/_base/act.yaml` | ACT 기본 하이퍼파라미터. 유일한 원본이며 모든 ACT 실험이 상속합니다. |
| `experiments/act_smoke.yaml` | 작은 데이터셋으로 100 step 실행. 전체 동작 확인용. |
| `experiments/peanut_act_80k.yaml` | 실제 peanut pick-and-place 학습. |
| `experiments/peanut_act_kl5.yaml` | 위와 동일하되 `kl_weight: 5.0`. |

### 패키지

| 파일 | 역할 |
|---|---|
| `cyclo_train/cli.py` | 명령줄 인터페이스. 모든 서브커맨드가 여기 있습니다. |
| `cyclo_train/config.py` | YAML 로드, `_base` 병합, `${…}` 치환, 검증, 해시. 학습이 쓰는 모든 경로를 소유합니다. |
| `cyclo_train/backends/base.py` | `TrainerBackend` 규약: 설정을 argv로 변환하는 것만 담당합니다. |
| `cyclo_train/backends/lerobot.py` | `lerobot-train` 명령어 생성. upstream을 감싸기만 하고 수정하지 않습니다. |
| `cyclo_train/runner.py` | 사전 검사, 학습 디렉터리 생성, manifest 기록, 출력 스트리밍(터미널 + 로그 동시). |
| `cyclo_train/manifest.py` | 이력 정보(git, 이미지, 데이터셋, GPU) 수집 → `manifest.json`. |
| `cyclo_train/promote.py` | 체크포인트를 추론용 위치로 복사, 학습 목록 조회. |
| `cyclo_train/data_prep/dataset.py` | 데이터셋 식별자·경로·`info.json`과 공용 22→19 트리밍 로직. |
| `cyclo_train/data_prep/download.py` | Hub 다운로드 + upstream v2.1→v3.0 변환. |
| `cyclo_train/data_prep/convert_v21.py` | 자체 v2.1→v3.0 변환기 (변환과 동시에 트리밍 가능). |
| `cyclo_train/data_prep/no_mobile.py` | v3.0 데이터셋을 22→19차원으로 트리밍. |
| `cyclo_train/analysis/loss.py` | 학습 로그 파싱 후 loss 수렴 구간 탐지. |

### 이 디렉터리 밖

| 파일 | 역할 |
|---|---|
| `docker/container.sh` | `train-lerobot` — 유일한 호스트 진입점. 컨테이너 실행, git 이력 주입, tmux 관리. |
| `docker/docker-compose.yml` | `lerobot_train` 서비스 (`train` 프로파일이라 자동 시작되지 않음). |
| `cyclo_brain/policy/lerobot/Dockerfile.train` | 학습 이미지: 서빙 이미지 + `wandb`. |
| `docker/.env` | `WANDB_API_KEY` 등. git에서 제외됨. |

---

## 개발

```bash
cd cyclo_brain/train
python -m pytest tests/ -q          # GPU·데이터셋·컨테이너 없이 실행됩니다
```

| 테스트 파일 | 범위 |
|---|---|
| `test_config_and_argv.py` | 설정 병합과 생성되는 학습 명령어. |
| `test_runner_and_promote.py` | 학습 디렉터리, 로그, manifest, 종료 코드, 배포. |
| `test_data_prep_and_analysis.py` | 트리밍 로직, 데이터셋 경로, 다운로드 argv, loss 분석. |
| `test_container_entrypoint.py` | `container.sh train-lerobot` 서브커맨드. |

유지해야 할 두 가지 제약:

- **import 시점에는 순수 Python(`pyyaml`)만 사용합니다.** 이 패키지는 이미지에
  읽기 전용으로 마운트되어 모든 이미지에서 수정 없이 import되어야 합니다.
  `numpy`/`pandas`는 `data_prep` 함수 안에서만 import하며, 모듈 최상단에서는
  import하지 않습니다. 이를 검증하는 테스트가 있습니다.
- **백엔드는 문자열만 생성합니다** — I/O도, 부수 효과도 없습니다. 그래야 GPU 없이
  테스트할 수 있습니다.

### 새 정책 백엔드 추가

1. `TrainerBackend`를 구현한 `cyclo_train/backends/<name>.py` 추가
2. 기본값을 담은 `experiments/_base/<name>.yaml` 추가
3. `backends/__init__.py`에서 import하여 등록
