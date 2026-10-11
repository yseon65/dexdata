# dexdata: XHand 궤적 데이터 형식과 수집기

DexGen 학습용 XHand 손 궤적을 **하나의 공통 형식**으로 모은다. 데이터 원천은 두 가지다.

| 원천 | env | 수집 스크립트 |
|---|---|---|
| dexscrew nut-bolt / screwdriver | `dexscrew/tasks/xhand_hora_lab.py` (`XHandHoraLab`) | `scripts/collect_dexscrew.py` |
| dexgen 자유 너트 | `dexgen/nut_goal_env.py` (`NutGoalEnv`), `nut_spin_env.py` (`NutSpinEnv`) 또는 이후의 새 env | `scripts/collect_dexgen.py` |

설계 근거는 [dexgen/docs/plan_review_and_revision.md](../dexgen/docs/plan_review_and_revision.md) 단계 1을 따른다.

- 모든 finger link를 저장하고, keypoint 집합(K)은 학습할 때 고른다.
- 손목 좌표계를 쓰고, quaternion은 wxyz로 통일한다.
- DexterityGen 논문의 정제 규칙(낙하로 끝난 episode는 마지막 2초를 버림)을 적용한다.

DexGen 모델과 학습 코드는 이 폴더에 없다. 이 폴더는 데이터 형식, 수집, 검증, 읽기만 담당한다.

**2026-10-11 기준 실제 수집 방법(사용할 정책, 필수 설정, 수집량, 시간)은 [docs/collection_guide.md](docs/collection_guide.md)를 본다.**

## 구조

```
dexdata/
  dexdata/schema.py      필드 정의, 규약, keypoint 목록, 종료 원인 코드 (형식의 단일 기준)
  dexdata/frames.py      wxyz quaternion·좌표계 변환 (torch/numpy 공용)
  dexdata/adapters.py    env → 공통 필드 (DexscrewAdapter, DexgenAdapter)
  dexdata/recorder.py    병렬 롤아웃 → episode 정제 → shard 저장, manifest 관리
  dexdata/dataset.py     읽기 (memory-map, episode 순회, history/horizon window 인덱스)
  scripts/collect_dexscrew.py
  scripts/collect_dexgen.py
  scripts/validate_dataset.py
  tests/test_frames.py
```

## 저장 형식 (schema v1.0)

```
<root>/manifest.json                              형식 정보, task/policy 이름표, 수집 run 기록
<root>/shards/<run>_<k>/<step_field>.npy          (T, *shape)  정책 step마다 1행, episode가 연속으로 저장됨
<root>/shards/<run>_<k>/ep_<episode_field>.npy    (E, *shape)  episode마다 1행 (start, length, 종료 원인, 물성)
```

### 규약

| 항목 | 규약 |
|---|---|
| 행 | 행 `t`는 행동 `a_t` 직전에 관측한 상태와 `a_t`를 담는다. 다음 상태는 같은 episode의 `t+1`행이다 (episode 마지막 행은 다음 상태 없음). |
| 시간 | 정책 step(20 Hz, `step_dt=0.05`)마다 1행. `manifest.runs[].step_dt` |
| 관절 순서 | `schema.JOINT_NAMES` (알파벳 순, dexscrew·dexgen과 같음) |
| Quaternion | **wxyz**, 단위 norm |
| 손목 좌표계 | 손 root link `left_hand_link`. `*_wrist` 필드는 모두 이 좌표계 기준 |
| 단위 | m, rad, s, N, N·m |
| 결측값 | float은 NaN, flag는 False (예: 자유 너트의 `object_dof_pos`, dexscrew의 손가락별 접촉) |

### step 필드 (자세한 내용은 `schema.STEP_FIELDS`)

| 그룹 | 필드 |
|---|---|
| proprio (DexGen 상태 입력) | `joint_pos`, `joint_vel`, `joint_target`, `control_error`, `keypoint_pos_wrist` (17,3), `keypoint_quat_wrist` (17,4) |
| 손목 | `wrist_pos_env`, `wrist_quat_world`, `gravity_wrist` |
| 행동 | `action`(정책 출력), `applied_target`(env noise와 limit을 거친 실제 목표), `applied_torque` |
| 물체 (privileged) | `object_pos_wrist`, `object_quat_wrist`, `object_linvel_wrist`, `object_angvel_wrist`, `object_dof_pos/vel`, `object_contact_force`, `finger_contact_force` (5) |
| 목표 | `goal_pos_wrist`, `goal_quat_wrist` (goal env에만 있음) |
| 신호 | `reward`, `success_event`, `goal_reached_event` |

keypoint 후보 17개는 다음과 같다. `schema.KEYPOINT_SETS`에 미리 정의한 집합이 있다.

- 손가락마다 link1(또는 bend link), link2, tip.
- 미리 정의한 집합: `tips`(K=5), `tips_mid`(K=10, 논문의 PIP+tip에 해당), `all`(K=17).

### episode 필드

`episode_id, env_id, task_id, policy_id, start, length, raw_length, end_reason, success, object_mass, object_scale, object_com_offset, friction, restitution, p_gain(12), d_gain(12)`

- `task_id`와 `policy_id`는 `manifest["tasks"]`, `manifest["policies"]`의 인덱스다.
- 물성 값은 그 episode가 진행되는 동안 쓰인 값이다.

### 정제 규칙 (`manifest.runs[].filter`)

| 규칙 | 기본값 | 이유 |
|---|---|---|
| episode 첫 행 제외 | `skip_first_steps=1` | reset 직후에는 link pose가 다음 physics step 전까지 갱신되지 않을 수 있다 |
| `drop` / `low_contact`로 끝난 episode의 마지막 2초 제거 | 40 step | 논문 Algorithm 1의 정제 규칙 |
| `timeout` / `stagnation` 종료 | 전부 보존 | 물체를 손에 쥔 상태라서 안전한 상태 분포에 속한다 |
| 정제 후 8행 미만 episode | 버림 | |
| 수집 종료 시 진행 중이던 episode | 버림 | |

종료 원인 코드(`schema.END_REASONS`)는 다음과 같다.

| 원천 | 원래 조건 | 저장되는 코드 |
|---|---|---|
| dexscrew | far (손가락이 nut에서 멂) | `drop` |
| dexscrew | no_contact | `low_contact` |
| dexscrew | stagnation | `stagnation` |
| dexscrew | joint limit | `other` |
| dexgen | dropped, lost, tipped (tilt > 45°) | `drop` |
| dexgen | 3-contact loss | `low_contact` |
| 공통 | time out | `timeout` |

## 사용법

```bash
cd ~/IsaacLab/projects/dexdata
PY=~/env_isaaclab6/bin/python

# 1) nut-bolt: dexscrew ProprioAdapt student (결정 D6). student 학습 조건(obs noise, reset 0.12)이 자동 적용된다.
$PY scripts/collect_dexscrew.py --out data/v1 --run_name nutbolt_student_s0 --num_envs 1024 \
    --target_transitions 3e7 --seed 0 \
    --ckpt ~/IsaacLab/projects/dexscrew/outputs/XHandHoraNutBolt_student_padapt/lab_run1/stage2_nn/model_last.ckpt

# 2) 자유 너트: dexgen goal env + RSL-RL 체크포인트 (env cfg는 --cfg key=value로 덮어쓰기 가능)
$PY scripts/collect_dexgen.py --out data/v1 --run_name nut_goal_s0 --num_envs 1024 --target_transitions 3e7 \
    --checkpoint ~/IsaacLab/projects/dexgen/outputs/xhand_nut_goal/<run>/model_XXXX.pt --stochastic \
    --cfg dr_enabled=True wrist_randomization_deg=30.0
#    다른 env: --env_class dexgen.nut_spin_env:NutSpinEnv --env_cfg dexgen.nut_spin_env:NutSpinEnvCfg

# 3) 검증 (schema, quaternion, 시간 정렬, URDF FK로 keypoint 재계산: 위치 1 mm, 자세 0.5° 기준)
$PY scripts/validate_dataset.py data/v1
```

- 같은 `--out`에 여러 run을 쌓을 수 있다. `--run_name`은 dataset 안에서 고유해야 한다.
- 정책이나 보상 스타일을 바꿔 수집할 때는 `--policy_name`으로 구분하면, 나중에 `TrajectoryDataset(policies=[...])`로 골라 쓸 수 있다.
- 기본 task 이름: dexscrew는 `nutbolt_<object>`, dexgen은 env class 이름이다. `--task_name`으로 바꿀 수 있다.

읽기:

```python
from dexdata.dataset import TrajectoryDataset
from dexdata.schema import KEYPOINT_SETS
ds = TrajectoryDataset("data/v1", tasks=["nutbolt_trinut", "NutGoalEnv"])
print(ds.summary())
idx = ds.window_index(history=4, horizon=4)        # (M, 2) [shard, row]: window가 episode 안에 들어가는 위치
w = ds.get_window(*idx[0], history=4, horizon=4, fields=["keypoint_pos_wrist", "joint_pos", "joint_target",
                                                          "control_error", "applied_target"])
kp = w["keypoint_pos_wrist"][:, KEYPOINT_SETS["tips_mid"]]   # (8, 10, 3)
```

참고로 논문은 정책 10 Hz에 T=2(0.2 s)를 쓴다. 이 데이터는 20 Hz이므로 2 step 간격으로 subsample하거나 horizon을 4로 잡으면 된다.

## 크기와 속도 (2026-10-09 측정)

- **행당 크기:** 244 float32, 약 0.95 KB. 1e8 transition이면 약 98 GB다.
- **메모리:** recorder는 CPU 메모리에 `num_envs × (max episode + 2) × 244 × 4 B`를 잡는다. 1024 env × 802 step이면 약 0.8 GB, 8192 env면 약 6.4 GB다.
- **속도:**
  - smoke test는 dexgen 학습과 GPU를 같이 쓰는 상태에서 16–32 env로 돌렸다. 이때 300–440 step/s가 나왔다.
  - 큰 env 수에서는 시뮬레이션 속도에 가깝게 나올 것으로 예상한다. 예를 들어 dexscrew teacher 학습은 8192 env에서 약 45k step/s였다.
  - 대규모 수집 처리량은 아직 측정하지 않았다.

## 검증 상태와 알려진 사항

- **dexscrew student(trinut)와 dexgen `NutSpinEnv` 체크포인트로 smoke 수집을 했다.** `validate_dataset.py`의 모든 검사를 통과했다.
  - FK 위치 오차는 0.001 mm 미만, 자세 오차는 평균 0.03°였다.
  - `joint_target[t+1] == applied_target[t]` 정렬은 정확히 일치했다.
- **`NutGoalEnv` 수집은 2026-10-11에 확인했다.** `validate_dataset.py` 모든 검사를 통과했다. 이전에 기록한 `_randomize_physics` IndexError는 env 쪽에서 해결됐다. 자세한 내용은 `docs/collection_guide.md` 6절을 본다.
- **dexscrew 쪽 특성**
  - `finger_contact_force`는 NaN이다. 손가락별 접촉 센서가 없다.
  - `object_*`는 회전하는 nut(handle) link의 pose다. bolt 축에 고정돼 있어서 위치는 일정하다.
  - `object_com_offset`의 z가 약 0.35인 것은 dexscrew 원본 코드의 COM 설정을 그대로 따른 것이다.
- **물성 분포가 원천마다 다르다.**
  - dexscrew 데이터는 dexscrew 원본 DR(마찰 0.5–8.0 등)로 수집된다. dexgen 데이터는 해당 env의 DR 범위를 따른다(결정 D2).
  - episode 필드에 실제 값이 저장되므로, 필요하면 범위로 필터링해서 쓰면 된다.
- **dexgen env의 attribute에 의존한다.** env를 고칠 때 아래 이름이 바뀌면 `adapters.DexgenAdapter`도 함께 맞춰야 한다.
  - `targets`, `effort`, `_pre_physics_step`
  - `nut_contact`, `finger_contacts` 센서
  - `extras["task/terminal_reason"]`, `extras["task/success"]`, `extras["task/goal_reached"]`
  - `goal_pos`, `goal_quat`, `p_factor`, `d_factor`, `scale`
