import os

dataprep_commands = 'cd FL-bench/data/utils\npython run.py -d tiny_imagenet -cn 100 -a 0.5'

print(dataprep_commands)
os.system(dataprep_commands)

mu = [0.0, 0.1, 0.3, 0.7, 1.0, 4.0] ## default: 1.0


for i in range(len(mu)):
    exp_commands = f'cd FL-bench/src/server\npython fedprox.py --model res18 -d tiny_imagenet --visible 0 -jr 0.4 --mu {mu[i]}'
    print(exp_commands)
    os.system(exp_commands)
    mv_command = f'mv FL-bench/out/FedProx/tiny_imagenet_acc_metrics.csv FL-bench/out/FedProx/tiny_imagenet_acc_metrics__exp4_{mu[i]}.csv'
    print(mv_command)
    os.system(mv_command)
# os.system('ls')