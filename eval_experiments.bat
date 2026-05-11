@echo off

@REM echo Running no-fault experiments with BASELINE controller, NORMAL controller, NORMAL-PURE controller, FAULT controller, and FAULT-PURE controller.

@REM echo BASELINE:
@REM python main.py --no-fault --exp-name baseline --no-op
@REM echo NORMAL:
@REM python main.py --no-fault --exp-name normal
@REM echo NORMAL-PURE:
@REM python main.py --no-fault --exp-name normal --pure
@REM echo FAULT:
@REM python main.py --no-fault --exp-name fault
@REM echo FAULT-PURE:
@REM python main.py --no-fault --exp-name fault --pure


echo Running fault-injected experiments with BASELINE controller, NORMAL controller, NORMAL-PURE controller, FAULT controller, and FAULT-PURE controller.

echo BASELINE:
python main.py --eval --exp-name baseline --no-op
echo NORMAL-PURE:
python main.py --eval --exp-name normal --pure
echo FAULT-PURE:
python main.py --eval --exp-name fault --pure
echo FINE-TUNE:
python main.py --eval --exp-name rl_7_ft_3 --pure --ft
echo NORMAL:
python main.py --eval --exp-name normal
echo FAULT:
python main.py --eval --exp-name fault


echo Running jittering-fault experiments with BASELINE controller, NORMAL controller, NORMAL-PURE controller, FAULT controller, and FAULT-PURE controller.

echo BASELINE:
python main.py --eval --exp-name baseline --no-op --jitter-fault
echo NORMAL-PURE:
python main.py --eval --exp-name normal --pure --jitter-fault
echo FAULT-PURE:
python main.py --eval --exp-name fault --pure --jitter-fault
echo FINE-TUNE:
python main.py --eval --exp-name rl_7_ft_3 --pure --ft --jitter-fault
echo NORMAL:
python main.py --eval --exp-name normal --jitter-fault
echo FAULT:
python main.py --eval --exp-name fault --jitter-fault


echo Running 2-wheeled damage experiments with BASELINE controller, NORMAL controller, NORMAL-PURE controller, FAULT controller, and FAULT-PURE controller.

echo BASELINE:
python main.py --eval --exp-name baseline --no-op --num-fault-wheels 2
echo NORMAL-PURE:
python main.py --eval --exp-name normal --pure --num-fault-wheels 2
echo FAULT-PURE:
python main.py --eval --exp-name fault --pure --num-fault-wheels 2
echo FINE-TUNE:
python main.py --eval --exp-name rl_7_ft_3 --pure --ft --num-fault-wheels 2
echo NORMAL:
python main.py --eval --exp-name normal --num-fault-wheels 2
echo FAULT:
python main.py --eval --exp-name fault --num-fault-wheels 2


@echo off
echo Running 3-wheeled damage experiments with BASELINE controller, NORMAL controller, NORMAL-PURE controller, FAULT controller, and FAULT-PURE controller.

echo BASELINE:
python main.py --eval --exp-name baseline --no-op --num-fault-wheels 3
echo NORMAL-PURE:
python main.py --eval --exp-name normal --pure --num-fault-wheels 3
echo FAULT-PURE:
python main.py --eval --exp-name fault --pure --num-fault-wheels 3
echo FINE-TUNE:
python main.py --eval --exp-name rl_7_ft_3 --pure --ft --num-fault-wheels 3
echo NORMAL:
python main.py --eval --exp-name normal --num-fault-wheels 3
echo FAULT:
python main.py --eval --exp-name fault --num-fault-wheels 3

echo Running 2-wheeled damage with jitter experiments with BASELINE controller, NORMAL controller, NORMAL-PURE controller, FAULT controller, and FAULT-PURE controller.

echo BASELINE:
python main.py --eval --exp-name baseline --no-op --num-fault-wheels 2 --jitter-fault
echo NORMAL-PURE:
python main.py --eval --exp-name normal --pure --num-fault-wheels 2 --jitter-fault
echo FAULT-PURE:
python main.py --eval --exp-name fault --pure --num-fault-wheels 2 --jitter-fault
echo FINE-TUNE:
python main.py --eval --exp-name rl_7_ft_3 --pure --ft --num-fault-wheels 2 --jitter-fault
echo NORMAL:
python main.py --eval --exp-name normal --num-fault-wheels 2 --jitter-fault
echo FAULT:
python main.py --eval --exp-name fault --num-fault-wheels 2 --jitter-fault


@echo off
echo Running 3-wheeled damage with jitter experiments with BASELINE controller, NORMAL controller, NORMAL-PURE controller, FAULT controller, and FAULT-PURE controller.

echo BASELINE:
python main.py --eval --exp-name baseline --no-op --num-fault-wheels 3 --jitter-fault
echo NORMAL-PURE:
python main.py --eval --exp-name normal --pure --num-fault-wheels 3 --jitter-fault
echo FAULT-PURE:
python main.py --eval --exp-name fault --pure --num-fault-wheels 3 --jitter-fault
echo FINE-TUNE:
python main.py --eval --exp-name rl_7_ft_3 --pure --ft --num-fault-wheels 3 --jitter-fault
echo NORMAL:
python main.py --eval --exp-name normal --num-fault-wheels 3 --jitter-fault
echo FAULT:
python main.py --eval --exp-name fault --num-fault-wheels 3 --jitter-fault

echo Generating graphs
cd eval_logs
python eval_grapher.py --experiment baseline=baseline-eval_log.csv --experiment normal_pure=normal_pure-eval_log.csv --experiment fault_pure=fault_pure-eval_log.csv --experiment fine_tune=rl_7_ft_3_pure_ft-eval_log.csv --experiment normal=normal-eval_log.csv --experiment fault=fault-eval_log.csv --no-show
echo Eval experiments finished.
