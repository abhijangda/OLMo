import os
import subprocess
import re
import time
import sys
import shutil

chkp_dir = sys.argv[1]

def slurp(p):
  with open(p, "r") as f:
    return f.read()

def run_command(c):
  with open("tmp.sh", "w") as f:
    f.write(c)

  process = subprocess.Popen(["bash", "-x", "tmp.sh"], stdout=subprocess.PIPE, text=True)

  for line in process.stdout:
      print(line, end="", flush=True)  # end="" prevents extra newlines

  process.wait()
    # print("Executing ", c, flush=True)
    # s,o = subprocess.getstatusoutput(c)
    # if s != 0:
    #     print("Error ", o, flush=True)

last_checkpoint_step = int(sys.argv[2])
intervals = int(sys.argv[3])
max_chkp = int(sys.argv[4])

chkp_idx = 0

def run():
  global last_checkpoint_step, chkp_idx
  BASE_CHKP_DIR = "/scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/OLMo-data/" + chkp_dir
  NODES = 8

  for node in range(NODES):
    shutil.copyfile(f"stdout{node}", os.path.join(BASE_CHKP_DIR, f"stdout{node}"))

  stdout0 = slurp("stdout0")
  steps = re.findall(r"Checkpoint saved to step(\d+)", stdout0)

  if len(steps) == 0:
    return

  latest_step = int(steps[-1])
  if not ((latest_step % intervals == 0 or latest_step % 1000 == 0) and latest_step > last_checkpoint_step):
    print(f"Step {latest_step} does not have checkpoint", flush=True)
    return

  last_checkpoint_step = latest_step

  OLMO_DIR="/home/aiscuser/ajangda/OLMo/"

  RUN_DIR = os.path.join(BASE_CHKP_DIR, f"latest_checkpoint_{chkp_idx % max_chkp}")
  chkp_idx += 1
  commands = ""

  if latest_step % 1000 != 0:
    if not os.path.exists(os.path.join(RUN_DIR, f"step{latest_step}")):
      for node in range(NODES):
        commands += f'ssh node-{node} "cd {OLMO_DIR}; zip -r0 latest_checkpoint.zip latest;" &\n'
      commands += "wait\n"
      for node in range(NODES):
        DST_DIR = os.path.join(RUN_DIR, f"node-{node}", )
        os.makedirs(DST_DIR, exist_ok = True)
        with open(os.path.join(RUN_DIR, f"step{latest_step}"), "w") as f:
          f.write("")
        commands += f'ssh node-{node} "cd {OLMO_DIR}; rsync -rP latest_checkpoint.zip {DST_DIR}" & \n'
        if node %2 == 1:
          commands += "wait\n"
    else:
      print("Checkpoint already exist\n")
      return

  if latest_step % 1000 == 0:
    RUN_DIR = BASE_CHKP_DIR

    for node in range(NODES):
      DST_DIR = os.path.join(RUN_DIR, f"node-{node}")
      os.makedirs(DST_DIR, exist_ok = True)

      commands += f'ssh node-{node} "cd {OLMO_DIR}; zip -r0 step{latest_step}.zip step{latest_step}; rsync -rP step{latest_step}.zip {DST_DIR}" & \n'
      if node % 2 == 1:
        commands += "wait\n"

  print(f"Saving checkpoint for step{latest_step} using: ", commands, flush=True)

  run_command(commands)
  print(f"Checkpoints saved")

while True:
  run()
  time.sleep(2)
