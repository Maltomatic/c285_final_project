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
python main.py --exp-name baseline --no-op
echo NORMAL:
python main.py --exp-name normal
echo NORMAL-PURE:
python main.py --exp-name normal --pure
echo FAULT:
python main.py --exp-name fault
echo FAULT-PURE:
python main.py --exp-name fault --pure

echo Generating graphs
cd eval_logs
python eval_grapher.py --experiment baseline=baseline-eval_log.csv --experiment normal_pure=normal_pure-eval_log.csv --experiment fault_pure=fault_pure-eval_log.csv --experiment normal=normal-eval_log.csv --experiment fault=fault-eval_log.csv --no-show
echo Eval experiments finished.
