#!/bin/bash
# Sequential DexGen data collection (dexdata/docs/collection_guide.md section 3)
cd ~/IsaacLab/projects/dexdata
PY=~/env_isaaclab6/bin/python
DG=~/IsaacLab/projects/dexgen/outputs/xhand_nut_goal
DS=~/IsaacLab/projects/dexscrew/outputs
OUT=data/v1
LOG=logs
echo "[a] start $(date)"
$PY scripts/collect_dexgen.py --out $OUT --run_name nut_goal_normal_s0 --seed 0 --num_envs 1024 \
    --target_transitions 4e7 --stochastic --policy_name nut_goal_normal \
    --checkpoint $DG/2026-10-10_23-26-04_goal_scale0713_ft_initq/model_14250.pt \
    --cfg "scale_range=(0.7,1.3)" dr_ramp_steps=0 > $LOG/a_nut_goal_normal_s0.log 2>&1
echo "[a] exit $? $(date)"; grep "complete" $LOG/a_nut_goal_normal_s0.log
echo "[b] start $(date)"
$PY scripts/collect_dexgen.py --out $OUT --run_name nut_goal_slow_s0 --seed 1 --num_envs 1024 \
    --target_transitions 2e7 --stochastic --policy_name nut_goal_slow15 \
    --checkpoint $DG/2026-10-11_03-53-58_goal_scale0713_ft_slowstyle15/model_13998.pt \
    --cfg "scale_range=(0.7,1.3)" dr_ramp_steps=0 > $LOG/b_nut_goal_slow_s0.log 2>&1
echo "[b] exit $? $(date)"; grep "complete" $LOG/b_nut_goal_slow_s0.log
echo "[c] start $(date)"
$PY scripts/collect_dexscrew.py --out $OUT --run_name nutbolt_student_s0 --seed 2 --num_envs 1024 \
    --target_transitions 4e7 --policy_name nutbolt_student_scale \
    --ckpt $DS/XHandHoraNutBolt_student_padapt/lab_scale07125_t1413/stage2_nn/model_best.ckpt \
    "task.env.randomization.randomizeScaleList=[0.7,0.8,0.9,1.0,1.1,1.2,1.25]" +task.env.keepNutHeight=True > $LOG/c_nutbolt_student_s0.log 2>&1
echo "[c] exit $? $(date)"; grep "complete" $LOG/c_nutbolt_student_s0.log
echo "[validate] start $(date)"
$PY scripts/validate_dataset.py $OUT > $LOG/validate_v1.log 2>&1
echo "[validate] exit $? $(date)"; grep -E "PASS|FAIL|ALL|summary" $LOG/validate_v1.log | tail -30
du -sh $OUT
