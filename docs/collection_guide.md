# 데이터 수집 가이드 (2026-10-11)

2026-10-10~11에 학습을 마친 정책들로 DexGen 학습 데이터를 모으는 방법을 정리한다. 학습 과정과 판단 근거는 [dexgen/docs/training_log_2026-10-10.md](../../dexgen/docs/training_log_2026-10-10.md)에 있다.

## 1. 수집에 쓸 정책

| 원천 | 정책 | checkpoint | 크기 범위 |
|---|---|---|---|
| dexgen 자유 너트, 일반 스타일 | D `goal_scale0713_ft_initq` | `dexgen/outputs/xhand_nut_goal/2026-10-10_23-26-04_goal_scale0713_ft_initq/model_14250.pt` | 0.7–1.3 연속 |
| dexgen 자유 너트, slow 스타일 (fingertip speed penalty 15) | `goal_scale0713_ft_slowstyle15` | `dexgen/outputs/xhand_nut_goal/2026-10-11_03-53-58_goal_scale0713_ft_slowstyle15/model_13998.pt` | 0.7–1.3 연속 |
| dexscrew nut-bolt (trinut), ProprioAdapt student (D6) | `lab_scale07125_t1413` | `dexscrew/outputs/XHandHoraNutBolt_student_padapt/lab_scale07125_t1413/stage2_nn/model_best.ckpt` | 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25 |

- 두 dexgen 정책은 모두 고정 크기 정책(`goal_contact2`, `goal_contact2_slowstyle15`)에서 0.7–1.3으로 fine-tune했다. 둘 다 bidirectional goal ±1.75 rad, upright goal과 tilt 처리, paper DR(D2)을 쓴다.
- dexscrew student의 teacher는 `XHandHoraNutBolt_teacher/lab_scale07125/stage1_nn/best_reward_1413.08.pth`다.
- 사용자 결정에 따라 dexscrew는 모양을 trinut 하나만 쓴다.
- dexscrew 크기 1.3은 접촉 solver가 폭주해서 학습과 수집에서 모두 뺐다. 리셋 직후 넛 joint가 1000 rad/s 이상 튄다.

### 크기별 성능 (명목 물리, 결정론적 정책)

**dexgen** (`measure_scale_feasibility.py`, 64 env, 800 step): timeout까지 버틴 비율 / episode당 목표 도달 횟수

| 크기 | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 | 1.15 | 1.2 | 1.25 | 1.3 |
|---|---|---|---|---|---|---|---|---|---|
| D (일반) | 98% / 259 | 91% / 245 | 95% / 233 | 98% / 223 | 63% / 134 | 57% / 106 | **12% / 25** | 35% / 62 | 63% / 113 |
| slow | 85% / 227 | 86% / 227 | 90% / 229 | 95% / 219 | 68% / 148 | – | **11% / 23** | – | 63% / 109 |
| 참고: `goal_contact2` (1.0 고정 학습) | 1% | 25% | 87% | 99% | 64% | – | 11% | – | 45% |

**dexscrew** (`eval_policy.py --student`, 256 env, 1000 step): rotation/episode (rad)

| 크기 | 0.7 | 0.85 | 1.0 | 1.15 | 1.2 | 1.25 |
|---|---|---|---|---|---|---|
| student `lab_scale07125_t1413` (사용) | 4.1 | 7.2 | 8.7 | 10.8 | 10.8 | 1.3 |
| student lab_run1 (0.95–1.05 학습) | 4.4 | 7.2 | 12.2 | 7.4 | 8.1 | 1.2 |
| teacher `best_reward_1413` (privileged) | 6.8 | 12.4 | 15.1 | 16.7 | 14.5 | 1.7 |

### 알려진 한계

- **dexgen 1.15–1.25 구간이 약하다.** 특히 1.2 근처가 나쁘다.
  - 두 정책 모두 1.2에서만 무너진다. 1.3은 오히려 낫다.
  - 그래서 정책보다 이 크기에서의 INIT_Q 초기 grasp가 원인으로 보인다.
  - v2 robust grasp 표로 보완하려던 시도(E run)는 실패했다. 이 grasp들이 22–38° 기울어 있어서 upright 목표에 도달하지 못했다.
  - 이 구간 데이터에는 drop으로 끝나는 짧은 episode가 많다. 낙하 직전 2초는 자동으로 제거된다.
  - 데이터 분포를 고르게 하려면 아래 4절처럼 크기 범위를 나눠서 수집량을 조절하면 된다.
- **dexscrew student는 크기가 섞이면 teacher 대비 성능이 떨어진다.**
  - 1.0에서 teacher 대비 58%다. lab_run1은 84%였다.
  - 1.0 근처의 회전 품질은 lab_run1 student가 더 좋다.
  - 필요하면 lab_run1 student 데이터를 `--policy_name`으로 구분해서 일부 섞어 쓸 수 있다 (4절 참고).
- **dexscrew 1.25는 거의 돌리지 못한다** (1.3 rad/episode). 대부분 stagnation 종료이고, 쥔 상태 데이터로는 남는다.

## 2. 반드시 지킬 설정

1. **dexgen에는 `--cfg dr_ramp_steps=0`을 꼭 준다.**
   - 이게 없으면 수집 env도 학습처럼 DR이 0부터 ramp된다. 그러면 첫 48000 env step 동안 질량, 마찰, 손목 회전(±30°, D3), 외력, noise가 모두 거의 명목값이다.
   - smoke test로 확인했다. 기본값에서는 질량 0.05–0.052, 마찰 1.0이었다. `dr_ramp_steps=0`을 주면 질량 0.031–0.239, 마찰 0.52–1.2가 된다.
2. **dexgen에는 `--cfg "scale_range=(0.7,1.3)"`을 준다.** env 기본값은 (0.95, 1.05)다.
3. **dexscrew에는 크기 override 두 개를 학습과 똑같이 준다.**
   - `"task.env.randomization.randomizeScaleList=[0.7,0.8,0.9,1.0,1.1,1.2,1.25]"`
   - `+task.env.keepNutHeight=True`
   - `keepNutHeight`가 없으면 작은 넛이나 큰 넛이 손 높이에서 벗어난다.
4. **student 학습 조건(obs noise 0.01/0.02, reset 0.12)은 `--preset auto`로 자동 적용된다.** 따로 줄 필요가 없다.
5. **run마다 다른 `--seed`와 고유한 `--run_name`을 쓴다.**
6. **정책과 스타일은 `--policy_name`으로 구분해 둔다.** 나중에 `TrajectoryDataset(policies=[...])`로 골라 쓸 수 있다.

## 3. 명령

```bash
cd ~/IsaacLab/projects/dexdata
PY=~/env_isaaclab6/bin/python
DG=~/IsaacLab/projects/dexgen/outputs/xhand_nut_goal
DS=~/IsaacLab/projects/dexscrew/outputs
OUT=data/v1

# (a) dexgen 일반 스타일
$PY scripts/collect_dexgen.py --out $OUT --run_name nut_goal_normal_s0 --seed 0 --num_envs 1024 \
    --target_transitions 4e7 --stochastic --policy_name nut_goal_normal \
    --checkpoint $DG/2026-10-10_23-26-04_goal_scale0713_ft_initq/model_14250.pt \
    --cfg "scale_range=(0.7,1.3)" dr_ramp_steps=0

# (b) dexgen slow 스타일
$PY scripts/collect_dexgen.py --out $OUT --run_name nut_goal_slow_s0 --seed 1 --num_envs 1024 \
    --target_transitions 2e7 --stochastic --policy_name nut_goal_slow15 \
    --checkpoint $DG/2026-10-11_03-53-58_goal_scale0713_ft_slowstyle15/model_13998.pt \
    --cfg "scale_range=(0.7,1.3)" dr_ramp_steps=0

# (c) dexscrew nut-bolt student
$PY scripts/collect_dexscrew.py --out $OUT --run_name nutbolt_student_s0 --seed 2 --num_envs 1024 \
    --target_transitions 4e7 --policy_name nutbolt_student_scale \
    --ckpt $DS/XHandHoraNutBolt_student_padapt/lab_scale07125_t1413/stage2_nn/model_best.ckpt \
    "task.env.randomization.randomizeScaleList=[0.7,0.8,0.9,1.0,1.1,1.2,1.25]" +task.env.keepNutHeight=True

# 검증 (schema, quaternion, 시간 정렬, URDF FK)
$PY scripts/validate_dataset.py $OUT
```

- **`--stochastic` (dexgen):** 학습 때 쓴 고정 std 0.3 Gaussian에서 행동을 sample한다. 같은 상태에서도 행동이 다양해진다. 결정론적 행동만 원하면 이 옵션을 뺀다.
- **명령 하나에 GPU 하나:** 여러 run은 순서대로 돌리는 것을 권한다. 동시에 돌리면 GPU 메모리(16 GB)가 부족할 수 있다. 1024 env 수집 하나는 학습보다 훨씬 가볍지만, dexgen env는 PhysX 버퍼를 크게 잡는다.
- **긴 수집 나누기:** `--run_name`을 `_s0`, `_s1`처럼 나눠 여러 번 돌려도 된다. 같은 `--out`에 쌓인다.

## 4. 수집량 (D5: 총 1e8 transition)

| 원천 | 권장 양 | 비율 |
|---|---|---|
| (a) dexgen 일반 | 4e7 | 40% |
| (b) dexgen slow | 2e7 | 20% |
| (c) dexscrew student | 4e7 | 40% |

- 이 비율은 제안이다. DexGen 학습에서 원천별 sampling 비율을 따로 조절할 수 있으므로, 수집 비율이 꼭 정확할 필요는 없다.
- **dexgen 크기 분포를 고르게 하고 싶을 때:**
  - 1.15–1.25 구간은 drop이 많아서, 정제 후 이 구간의 비중이 작아진다.
  - 크기 범위를 나눠 두 run으로 수집하면 비중을 맞출 수 있다. 예: `scale_range=(0.7,1.1)` 3e7, `scale_range=(1.1,1.3)` 1e7 추가.
  - 크기는 episode 필드 `object_scale`에 저장되므로, 수집 후 크기별 비율을 확인할 수 있다.
- **dexscrew 1.0 근처 품질을 보강하고 싶을 때:** lab_run1 student를 `--policy_name nutbolt_student_run1`으로 1e7 정도 추가한다. 이때 크기 override는 기본값(0.95/1.0/1.05)으로 둔다.

## 5. 시간, 용량, 수율 (1024 env 측정, 다른 GPU 작업 없음)

| 원천 | raw 처리량 | 정제 후 남는 비율 | 4e7 kept에 필요한 시간 |
|---|---|---|---|
| dexgen (full DR) | 16.9k step/s | 59% (drop 많음) | 약 67분 |
| dexscrew student | 16.4k step/s | 83% (stagnation과 timeout은 전부 보존) | 약 49분 |

- **용량:** 행당 약 0.95 KB다. 1e8 transition이면 약 98 GB이므로 디스크 여유를 먼저 확인한다.
- **수율:**
  - 64 env smoke test에서 dexgen full DR의 수율은 39%였다. env 수가 많을수록 수집이 끝날 때 버려지는 진행 중 episode의 비율이 줄어든다.
  - dexgen에서 drop으로 끝난 episode는 마지막 40 step(2초)이 잘린다. 그 뒤 8행 미만이면 episode 전체를 버린다.
- **CPU 메모리:** recorder는 1024 env 기준 약 0.8 GB를 쓴다.

## 6. smoke test로 확인한 것 (2026-10-11)

- **`NutGoalEnv` 수집 확인.** README의 "아직 수집하지 못함"은 이제 해당하지 않는다. 그 사이 env의 `_randomize_physics`가 정상 동작하도록 바뀌었다.
  - goal 필드(`goal_*_wrist`, `goal_reached_event`)도 저장된다.
  - `p_factor`, `d_factor`, `scale` 읽기도 확인했다.
- **새 크기 설정의 dexgen D와 dexscrew student 데이터가 `validate_dataset.py` 모든 검사를 통과했다.**
  - FK keypoint 위치: 0.000 mm
  - FK keypoint 자세: 평균 0.03°, 최대 0.10°
  - `joint_target[t+1] == applied_target[t]` 정렬: 정확히 일치
- **episode 필드 `object_scale` 확인**
  - dexgen: 0.71–1.30 연속값
  - dexscrew: 7개 버킷 값

## 7. 실제 수집 결과 (`data/v1`, 2026-10-11 09:01–10:51)

`logs/collect_v1.sh`로 3절의 명령 (a), (b), (c)를 차례로 실행했다. 각 run의 로그는 `logs/*_s0.log`, 검증 결과는 `logs/validate_v1.log`에 있다.

| run | `policy_name` | transition | episode | raw → 정제 후 | 시간 |
|---|---|---|---|---|---|
| `nut_goal_normal_s0` | `nut_goal_normal` | 40.0M | 169,736 | 45.9M → 87% | 45분 |
| `nut_goal_slow_s0` | `nut_goal_slow15` | 20.0M | 96,158 | 23.9M → 84% | 24분 |
| `nutbolt_student_s0` | `nutbolt_student_scale` | 40.0M | 143,115 | 40.4M → 99% | 41분 |
| 합계 | | **100.0M**, 91 GB | 409,009 | | 1시간 50분 |

- **검증:** `validate_dataset.py` 모든 검사를 통과했다.
  - FK keypoint 위치 최대 0.001 mm, 자세 최대 0.1°
  - 시간 정렬은 정확히 일치한다.
- **물성:** dexgen 질량은 0.03–0.25다. `dr_ramp_steps=0`으로 full DR이 적용된 것을 확인했다.
- **크기 분포 (정제 후 transition 비율)**
  - dexgen 일반: 0.7–0.85 25%, 0.85–1.05 33%, 1.05–1.15 16%, 1.15–1.22 11%, 1.22–1.3 14%
  - dexgen slow: 비슷한 분포
  - dexscrew: 0.7–0.85 29%, 0.85–1.05 29%, 1.05–1.15 14%, 1.15–1.22 14%, 1.25 14%
  - 거의 균등 분포라서 4절의 크기 범위 분할 수집은 하지 않았다.
- **종료 원인**
  - dexgen: drop이 많다. 일반 정책은 drop 379k, timeout 67k다. drop으로 끝난 episode는 마지막 2초가 잘렸고, 너무 짧아진 episode는 버렸다.
  - dexscrew: stagnation 112k, timeout 32k다. 넛을 쥔 상태라 전부 보존했다.
- **통계와 그림:** 정책별 관절, fingertip, 물체, coverage 분포는 [dataset_stats.md](dataset_stats.md)를 본다.
