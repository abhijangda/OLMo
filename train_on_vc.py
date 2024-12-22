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

if nodes > 1:
    run("rm -rf /home/aiscuser/ajangda/OLMo/step*")
    run("rm -rf /home/aiscuser/ajangda/OLMo/train_data/")
    run("cd /home/aiscuser/ajangda/ ; rm -rf OLMo.zip ; zip -r OLMo.zip /home/aiscuser/ajangda/OLMo/")
    for node in range(nodes):
        if node != 0:
            run(f"ssh node-{node} 'rm -rf /home/aiscuser/ajangda ; mkdir /home/aiscuser/ajangda'")
            run(f"scp /home/aiscuser/ajangda/OLMo.zip node-{node}:/home/aiscuser/ajangda/OLMo.zip")
            run(f"ssh node-{node} 'unzip -o /home/aiscuser/ajangda/OLMo.zip -d /'")
            run(f"ssh node-{node} 'cd /home/aiscuser/ajangda/OLMo/ ; pip install -e .[all] ; pip install ./pyfastkron-1.0.1-py3-none-any.whl'; pip install aioshutil")
        # run(f"ssh node-{node} killall -SIGKILL /home/ajangda/anaconda3/envs/mscclpp/bin/python")

train_id = datetime.now().strftime('%Y%m-%d%H-%M%S')
torchrun = f"torchrun --rdzv_id=12345 --rdzv_backend=c10d --rdzv_endpoint={ip}:8645 --nnodes {nodes} --nproc_per_node=8 scripts/train.py configs/official-1124/OLMo2-7B-stage1.yaml --save_overwrite --wandb.name={train_id}"
for node in range(nodes):
    olmo_data = f"/scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/OLMo-data/{train_id}"
    remote_folder=f"{olmo_data}/node-{node}/"
    os.makedirs(remote_folder, exist_ok=True)
    stdout=f"{remote_folder}/stdout"
    nohup_wrap = f"cd /home/aiscuser/ajangda/OLMo ; nohup {torchrun} --remote_save_folder={remote_folder} &> {stdout} &"
    run(f"ssh node-{node} '{nohup_wrap}'")

