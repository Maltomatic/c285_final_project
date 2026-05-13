@echo off

echo Running no-fault experiments with BASELINE controller, NORMAL controller, NORMAL-PURE controller, FAULT controller, and FAULT-PURE controller.

echo BASELINE:
python main.py --no-fault --exp-name baseline --no-op --eval
echo NORMAL-PURE:
python main.py --no-fault --exp-name normal --pure --eval
echo FAULT-PURE:
python main.py --no-fault --exp-name fault --pure --eval
echo FINE-TUNE:
python main.py --no-fault --exp-name rl_7_ft_3 --pure --ft --eval
echo NORMAL:
python main.py --no-fault --exp-name normal --eval
echo FAULT:
python main.py --no-fault --exp-name fault --eval
echo FAULT-H1:
python main.py --no-fault --exp-name fault --obs-stack 1 --eval 
echo FAULT-H3:
python main.py --no-fault --exp-name fault --obs-stack 3 --eval 
echo FAULT-H7:
python main.py --no-fault --exp-name fault --obs-stack 7 --eval 
echo FAULT-H10:
python main.py --no-fault --exp-name fault --obs-stack 10 --eval 


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
echo FAULT-H1:
python main.py --eval --exp-name fault --obs-stack 1
echo FAULT-H3:
python main.py --eval --exp-name fault --obs-stack 3
echo FAULT-H7:
python main.py --eval --exp-name fault --obs-stack 7
echo FAULT-H10:
python main.py --eval --exp-name fault --obs-stack 10


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
echo FAULT-H1:
python main.py --eval --exp-name fault --jitter-fault --obs-stack 1
echo FAULT-H3:
python main.py --eval --exp-name fault --jitter-fault --obs-stack 3
echo FAULT-H7:
python main.py --eval --exp-name fault --jitter-fault --obs-stack 7
echo FAULT-H10:
python main.py --eval --exp-name fault --jitter-fault --obs-stack 10


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
echo FAULT-H1:
python main.py --eval --exp-name fault --num-fault-wheels 2 --obs-stack 1
echo FAULT-H3:
python main.py --eval --exp-name fault --num-fault-wheels 2 --obs-stack 3
echo FAULT-H7:
python main.py --eval --exp-name fault --num-fault-wheels 2 --obs-stack 7
echo FAULT-H10:
python main.py --eval --exp-name fault --num-fault-wheels 2 --obs-stack 10


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
echo FAULT-H1:
python main.py --eval --exp-name fault --num-fault-wheels 3 --obs-stack 1
echo FAULT-H3:
python main.py --eval --exp-name fault --num-fault-wheels 3 --obs-stack 3
echo FAULT-H7:
python main.py --eval --exp-name fault --num-fault-wheels 3 --obs-stack 7
echo FAULT-H10:
python main.py --eval --exp-name fault --num-fault-wheels 3 --obs-stack 10


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
echo FAULT-H1:
python main.py --eval --exp-name fault --num-fault-wheels 2 --jitter-fault --obs-stack 1
echo FAULT-H3:
python main.py --eval --exp-name fault --num-fault-wheels 2 --jitter-fault --obs-stack 3
echo FAULT-H7:
python main.py --eval --exp-name fault --num-fault-wheels 2 --jitter-fault --obs-stack 7
echo FAULT-H10:
python main.py --eval --exp-name fault --num-fault-wheels 2 --jitter-fault --obs-stack 10


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
echo FAULT-H1:
python main.py --eval --exp-name fault --num-fault-wheels 3 --jitter-fault --obs-stack 1
echo FAULT-H3:
python main.py --eval --exp-name fault --num-fault-wheels 3 --jitter-fault --obs-stack 3
echo FAULT-H7:
python main.py --eval --exp-name fault --num-fault-wheels 3 --jitter-fault --obs-stack 7
echo FAULT-H10:
python main.py --eval --exp-name fault --num-fault-wheels 3 --jitter-fault --obs-stack 10


echo Generating graphs
cd eval_logs

echo Generating baseline-1 fault eval graphs
python eval_grapher.py --experiment baseline=baseline-eval_log.csv --experiment normal_pure=normal_pure-eval_log.csv --experiment fault_pure=fault_pure-eval_log.csv --experiment fine_tune=rl_7_ft_3_pure_ft-eval_log.csv --experiment normal=normal-eval_log.csv --experiment fault=fault-eval_log.csv --experiment fault_k1=fault_k1-eval_log.csv --experiment fault_k3=fault_k3-eval_log.csv --experiment fault_k7=fault_k7-eval_log.csv --experiment fault_k10=fault_k10-eval_log.csv --logs-dir exp --no-show

echo Generating baseline-no fault eval graphs
python multi_exp_eval_grapher.py --experiment-name nofault --csv baseline_nofault-eval_log.csv --csv normal_pure_nofault-eval_log.csv --csv fault_pure_nofault-eval_log.csv --csv fine_tune=rl_7_ft_3_pure_ft_nofault-eval_log.csv --csv normal_nofault-eval_log.csv --csv fault_nofault-eval_log.csv --csv fault_k1_nofault-eval_log.csv --csv fault_k3_nofault-eval_log.csv --csv fault_k7_nofault-eval_log.csv --csv fault_k10_nofault-eval_log.csv --no-show

echo Generating jitter comparison graph
python multi_exp_eval_grapher.py --experiment-name jitter --csv baseline_jitter-eval_log.csv --csv normal_pure_jitter-eval_log.csv --csv fault_pure_jitter-eval_log.csv --csv fine_tune=rl_7_ft_3_pure_ft_jitter-eval_log.csv --csv normal_jitter-eval_log.csv --csv fault_jitter-eval_log.csv --csv fault_k1_jitter-eval_log.csv --csv fault_k3_jitter-eval_log.csv --csv fault_k7_jitter-eval_log.csv --csv fault_k10_jitter-eval_log.csv --no-show

echo Generating 2-fault comparison graph
python multi_exp_eval_grapher.py --experiment-name 2faults --csv baseline_2faults-eval_log.csv --csv normal_pure_2faults-eval_log.csv --csv fault_pure_2faults-eval_log.csv --csv fine_tune=rl_7_ft_3_pure_ft_2faults-eval_log.csv --csv normal_2faults-eval_log.csv --csv fault_2faults-eval_log.csv --csv fault_k1_2faults-eval_log.csv --csv fault_k3_2faults-eval_log.csv --csv fault_k7_2faults-eval_log.csv --csv fault_k10_2faults-eval_log.csv --no-show

echo Generating 3-fault comparison graph
python multi_exp_eval_grapher.py --experiment-name 3faults --csv baseline_3faults-eval_log.csv --csv normal_pure_3faults-eval_log.csv --csv fault_pure_3faults-eval_log.csv --csv fine_tune=rl_7_ft_3_pure_ft_3faults-eval_log.csv --csv normal_3faults-eval_log.csv --csv fault_3faults-eval_log.csv --csv fault_k1_3faults-eval_log.csv --csv fault_k3_3faults-eval_log.csv --csv fault_k7_3faults-eval_log.csv --csv fault_k10_3faults-eval_log.csv --no-show

echo Generating jitter_2faults comparison graph
python multi_exp_eval_grapher.py --experiment-name jitter_2faults --csv baseline_jitter_2faults-eval_log.csv --csv normal_pure_jitter_2faults-eval_log.csv --csv fault_pure_jitter_2faults-eval_log.csv --csv fine_tune=rl_7_ft_3_pure_ft_jitter_2faults-eval_log.csv --csv normal_jitter_2faults-eval_log.csv --csv fault_jitter_2faults-eval_log.csv --csv fault_k1_jitter_2faults-eval_log.csv --csv fault_k3_jitter_2faults-eval_log.csv --csv fault_k7_jitter_2faults-eval_log.csv --csv fault_k10_jitter_2faults-eval_log.csv --no-show

echo Generating jitter_3faults comparison graph
python multi_exp_eval_grapher.py --experiment-name jitter_3faults --csv baseline_jitter_3faults-eval_log.csv --csv normal_pure_jitter_3faults-eval_log.csv --csv fault_pure_jitter_3faults-eval_log.csv --csv fine_tune=rl_7_ft_3_pure_ft_jitter_3faults-eval_log.csv --csv normal_jitter_3faults-eval_log.csv --csv fault_jitter_3faults-eval_log.csv --csv fault_k1_jitter_3faults-eval_log.csv --csv fault_k3_jitter_3faults-eval_log.csv --csv fault_k7_jitter_3faults-eval_log.csv --csv fault_k10_jitter_3faults-eval_log.csv --no-show

echo Eval experiments finished.
