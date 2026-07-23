# eval_tools — L2 채점 도구 + L3 집계 스크립트

이미 녹화된 LeRobot eval 데이터셋을 **사람이 키보드로 빠르게 채점(L2)** 하고,
그 결과를 **해석 가능한 통계로 집계(L3)** 하는 순수 로컬 도구입니다.
로봇/GPU/ROS2 의존성이 전혀 없고, 원본 데이터셋은 **읽기 전용**으로만 참조합니다.

```
eval_tools/
├── requirements.txt
├── l2_scoring/          # 채점 웹앱 (FastAPI + 브라우저)
│   ├── app.py           # 백엔드 엔트리포인트
│   ├── dataset.py       # 폴더명 정규식 파싱 + meta/episodes 읽기
│   ├── storage.py       # CSV 영속 저장 (세션 재개)
│   ├── video_merge.py   # 다중 카메라 → 단일 동기화 영상 (ffmpeg)
│   ├── failure_codes.py # F0~F6 / y_position 라벨 상수 (여기만 고치면 됨)
│   └── static/index.html
└── l3_aggregate/
    └── aggregate.py     # 집계 + matplotlib 그림
```

## 설치 (한 번만)

`uv`로 격리된 venv를 만듭니다 (시스템/conda base를 건드리지 않음):

```bash
cd eval_tools
uv venv
uv pip install -r requirements.txt
```

> `video_merge.py`는 시스템 `ffmpeg`가 있으면 그걸 쓰고, 없으면
> `imageio-ffmpeg`가 번들한 정적 ffmpeg 바이너리를 자동으로 사용합니다.
> 즉 시스템에 ffmpeg를 따로 설치할 필요가 없습니다.

## L2 — 채점 실행

```bash
cd eval_tools
uv run python -m l2_scoring.app --csv results/annotations.csv
# 또는 폴더를 미리 지정:
uv run python -m l2_scoring.app --eval-root /path/to/policy_A_step10000_chunk50 \
    --csv results/annotations.csv
```

브라우저에서 `http://127.0.0.1:8000` 접속 → 상단에 eval 폴더 경로 입력 후 **Load**.

- `policy_id / checkpoint_step / chunk_size`: 폴더명(`policy_A_step10000_chunk50`)에서
  정규식으로 자동 추출되며, 상단 필드에서 직접 수정 가능합니다.
- **다중 카메라 동기화**: 백엔드가 ffmpeg로 여러 카메라 뷰를 하나의 영상(hstack/vstack)으로
  병합해 스트리밍하므로 프레임 엇갈림이 없습니다. (상단 `layout`으로 가로/세로 전환)

### 키보드 단축키

| 키 | 동작 |
|---|---|
| `←` / `→` | 이전 / 다음 에피소드 (이동 시 자동 저장) |
| `S` / `F` | 성공 / 실패 |
| `0`~`6` | 실패 코드 F0~F6 (자동으로 실패 처리) |
| `Z` / `X` / `C` | y_position = top / mid / bottom |
| `Space` | 재생 / 일시정지 |
| `Enter` | 저장 후 다음 |
| `Esc` | 입력창 포커스 해제 |

채점 결과는 매번 CSV에 자동 저장되고, 같은 `run_id`(=폴더명) + `session_id`로 앱을
다시 열면 이전 상태가 복원됩니다.

### CSV 스키마

```
run_id, session_id, policy_id, checkpoint_step, chunk_size,
trial_index, init_condition_id, y_position, result, failure_code,
notes, video_path, annotated_by, annotated_at
```

- `run_id` = eval 폴더명, `session_id` = 채점 세션 식별자(상단에서 입력).
- 노이즈 바닥 측정: 같은 policy를 서로 다른 `session_id`로 두 번 채점하세요.

## L3 — 집계 실행

```bash
cd eval_tools
uv run python l3_aggregate/aggregate.py results/annotations.csv --outdir l3_out
# 여러 CSV / 정책 지정:
uv run python l3_aggregate/aggregate.py a.csv b.csv \
    --policy-a policy_A --policy-b policy_B --heatmap-dim y_position
```

출력:
1. **노이즈 바닥** — 같은 `policy_id` × 다른 `session_id` 성공률 절대차.
2. **조건별 성공률 + Wilson 95% CI** — `y_position × chunk_size × checkpoint_step`,
   셀 `n<8` 경고 플래그.
3. **McNemar 검정** — `init_condition_id` 기준 두 정책 inner join, 누락 조건 수 경고 로그,
   `b/c/b+c/p-value`, `b+c<10` 시 검정력 부족 경고.
4. **checkpoint 곡선** — step vs 성공률 + CI band (전체 곡선 + last-5 max) → PNG.
5. **F코드 × 조건 히트맵** → PNG.

## 라벨 커스터마이즈

`l2_scoring/failure_codes.py`의 `FAILURE_CODES` / `Y_POSITIONS` 상수만 수정하면
프론트/백엔드에 자동 반영됩니다 (HTML 수정 불필요).

## 하지 않는 것 (비목표)

- `control_robot.py` / record 파이프라인 / ROS2 / FFW 드라이버 수정하지 않음.
- 정책 재학습, 원본 데이터셋 변형하지 않음 (읽기 전용).
- success 자동판정(classifier/VLM) 없음 — **사람이 직접 채점**.
