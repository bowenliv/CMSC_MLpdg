import os

dataprep_commands = 'cd FL-bench/data/utils\npython run.py -d tiny_imagenet -cn 100 -a 0.5'

print(dataprep_commands)
os.system(dataprep_commands)

le = [1, 5, 10, 50, 100, 500] # default: 5
ge = [500, 100, 50, 10, 5, 1] # default: 100

for i in range(len(le)):
    exp_commands = f'cd FL-bench/src/server\npython fedavg.py --model res18 -d tiny_imagenet --visible 0 -jr 0.4 -le {le[i]} -ge {ge[i]}'
    print(exp_commands)
    os.system(exp_commands)
    mv_command = f'mv FL-bench/out/FedAvg/tiny_imagenet_acc_metrics.csv FL-bench/out/FedAvg/tiny_imagenet_acc_metrics__exp3_{le[i]}_{ge[i]}.csv'
    print(mv_command)
    os.system(mv_command)
# os.system('ls')