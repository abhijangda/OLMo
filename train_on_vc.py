import subprocess
from datetime import datetime
import sys
import os

nodes = int(sys.argv[1])
ip = sys.argv[2]

def run(c):
    print("Executing ", c)
    s,o = subprocess.getstatusoutput(c)
    if s != 0:
        print("Error ", o)

if os.path.exists("/ddn/ajangda/ssh-config"):
    SSH = "ssh -F /ddn/ajangda/ssh-config"
    SCP = "scp -F /ddn/ajangda/ssh-config"
else:
    SSH = "ssh"
    SCP = "scp"

if nodes > 1:
    run("rm -rf ~/ajangda/OLMo/step*")
    run("rm -rf ~/ajangda/OLMo/train_data/")
    run("rm -rf ~/ajangda/OLMo/latest/")
    run("cd ~/ajangda/ ; rm -rf OLMo.zip ; zip -r OLMo.zip ~/ajangda/OLMo/")
    commands = ""
    for node in range(nodes):
        if node != 0:
            commands += f"{SSH} node-{node} 'rm -rf ~/ajangda ; mkdir ~/ajangda' \n"
    
    for node in range(nodes):
        if node != 0:
            commands += f"{SCP} ~/ajangda/OLMo.zip node-{node}:~/ajangda/OLMo.zip &\n"
    commands += "wait\n"
    for node in range(nodes):
        if node != 0:
            commands += f"{SSH} node-{node} 'unzip -o ~/ajangda/OLMo.zip -d /' &\n"
    commands += "wait\n"
    for node in range(nodes):
        commands += f"{SSH} node-{node} 'cd ~/ajangda/OLMo/ ; pip install -e .[all] ; pip install ./pyfastkron-1.0.1-py3-none-any.whl; pip install aioshutil' &\n"
    
    run(commands)
    
        # run(f"ssh node-{node} killall -SIGKILL /home/ajangda/anaconda3/envs/mscclpp/bin/python")
#sys.exit(0)
#for node in range(nodes):
#    run(f"ssh node-{node} 'cd ~/ajangda/OLMo/ ; pip install -e .[all] ; pip install ./pyfastkron-1.0.1-py3-none-any.whl; pip install aioshutil'")

export_envs = "export NCCL_PROTO=Simple ; export NCCL_ALGO=Ring ; export TORCH_INDUCTOR_BACKEND=cpp ;" #"export NCCL_P2P_DISABLE=WARN ; export TORCH_DISTRIBUTED_DEBUG=DETAIL ; export TORCH_CPP_LOG_LEVEL=INFO; "
step = sys.argv[4]
train_id = sys.argv[3] #"202412-2110-2852-7B-compressed-run-03" #datetime.now().strftime('%Y%m-%d%H-%M%S')+"-7B-compressed"
import random
rdzv_id = random.randint(1000, 9999)
port = random.randint(10000,65536)
torchrun = f"torchrun --rdzv_id={rdzv_id} --rdzv_backend=c10d --rdzv_endpoint={ip}:{port} --nnodes {nodes} --nproc_per_node=8 scripts/train.py configs/official-1124/OLMo2-{train_id}.yaml --save_overwrite --wandb.name={train_id} --fsdp.wrapping_strategy=null --save_interval_ephemeral=100"
load_path_node_1 = f"~/mnt-node-0/ajangda/step{step}/"
load_path_node_0 = f"~/ajangda/step{step}/"
for node in range(nodes):
    olmo_data = f"/scratch/whitneyblobstore/OLMo-data/{train_id}"
    remote_folder=f"{olmo_data}/node-{node}/"
    os.makedirs(remote_folder, exist_ok=True)
    stdout=("stdout" if node==0 else "~/mnt-node-0/ajangda/OLMo/stdout")+ str(node) #f"{remote_folder}/stdout"
    load_path = load_path_node_0 if node == 0 else load_path_node_1
    load_path = f"--load_path={load_path}"
    nohup_wrap = f"cd ~/ajangda/OLMo ; nohup {torchrun}  &>> {stdout} &"
    run(f"{SSH} node-{node} '{export_envs} {nohup_wrap}'")