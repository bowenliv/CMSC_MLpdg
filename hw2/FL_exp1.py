import os


## part1-- different cn
cn = [20, 70, 100, 200] # default: 20; in given sample cmd: 100

for i in range(len(cn)):
    dataprep_commands = f'cd FL-bench/data/utils\npython run.py -d tiny_imagenet --iid 1 -cn {cn[i]}'
    print(dataprep_commands)
    os.system(dataprep_commands)
    exp_commands = f'cd FL-bench/src/server\npython fedavg.py --model res18 -d tiny_imagenet --visible 0 -jr 0.4'
    print(exp_commands)
    os.system(exp_commands)
    mv_command = f'mv FL-bench/out/FedAvg/tiny_imagenet_acc_metrics.csv FL-bench/out/FedAvg/tiny_imagenet_acc_metrics__exp1_part1_{cn[i]}.csv'
    print(mv_command)
    os.system(mv_command)
# os.system('ls')


## part2-- different jr
jr = [0.1, 0.4, 0.7, 1.0] # default: 0.1; in given sample cmd: .4

for i in range(len(jr)):
    dataprep_commands = f'cd FL-bench/data/utils\npython run.py -d tiny_imagenet --iid 1 -cn 100'
    print(dataprep_commands)
    os.system(dataprep_commands)
    exp_commands = f'cd FL-bench/src/server\npython fedavg.py --model res18 -d tiny_imagenet --visible 0 -jr {jr[i]}'
    print(exp_commands)
    os.system(exp_commands)
    mv_command = f'mv FL-bench/out/FedAvg/tiny_imagenet_acc_metrics.csv FL-bench/out/FedAvg/tiny_imagenet_acc_metrics__exp1_part2_{jr[i]}.csv'
    print(mv_command)
    os.system(mv_command)
# os.system('ls')