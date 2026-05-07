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

@echo off
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

echo Generating graphs
cd eval_logs
python eval_grapher.py --experiment baseline=baseline-eval_log.csv --experiment normal_pure=normal_pure-eval_log.csv --experiment fault_pure=fault_pure-eval_log.csv --experiment fine_tune=rl_7_ft_3_pure_ft-eval_log.csv --experiment normal=normal-eval_log.csv --experiment fault=fault-eval_log.csv --no-show
echo Eval experiments finished.
