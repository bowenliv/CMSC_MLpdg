import os


## part1
alpha = [0.1, 0.5, 1.0, 4.0] # default: 

for i in range(len(alpha)):
    dataprep_commands = f'cd FL-bench/data/utils\npython run.py -d tiny_imagenet -cn 100 -a {alpha[i]}'
    print(dataprep_commands)
    os.system(dataprep_commands)
    exp_commands = f'cd FL-bench/src/server\npython fedavg.py --model res18 -d tiny_imagenet --visible 0 -jr 0.4'
    print(exp_commands)
    os.system(exp_commands)
    mv_command = f'mv FL-bench/out/FedAvg/tiny_imagenet_acc_metrics.csv FL-bench/out/FedAvg/tiny_imagenet_acc_metrics__exp2_part1_{alpha[i]}.csv'
    print(mv_command)
    os.system(mv_command)
# os.system('ls')

## part2
dataprep_commands = f'cd FL-bench/data/utils\npython run.py -d tiny_imagenet -cn 100 -a 0.01'
print(dataprep_commands)
os.system(dataprep_commands)
exp_commands = f'cd FL-bench/src/server\npython fedavg.py --model res18 -d tiny_imagenet --visible 0 -jr 0.4'
print(exp_commands)
os.system(exp_commands)
mv_command = f'mv FL-bench/out/FedAvg/tiny_imagenet_acc_metrics.csv FL-bench/out/FedAvg/tiny_imagenet_acc_metrics__exp2_part2.csv'
print(mv_command)
os.system(mv_command)

