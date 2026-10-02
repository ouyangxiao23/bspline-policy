from pathlib import Path
import os,sys,json,time,subprocess,fcntl
ROOT=Path(__file__).resolve().parents[1]
PYTHON=os.environ.get('EVAL_PYTHON',sys.executable)
(ROOT/'reports').mkdir(parents=True,exist_ok=True)
STATE=ROOT/'reports/pair_status.json'
def save(stage,**kwargs):
 temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(dict(time=time.time(),launcher_pid=os.getpid(),stage=stage,**kwargs),indent=2));temp.replace(STATE)
def env(gpu):
 e=os.environ.copy();e.update(PYTHONPATH=os.environ.get('MICROWAVE_DEPS','/tmp/bsp-microwave-deps')+os.pathsep+e.get('PYTHONPATH',''),CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_MODE='offline',PYTHONUNBUFFERED='1',HYDRA_FULL_ERROR='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
 return e
lock=(ROOT/'reports/pair.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for model,gpu in [('dense',0),('bsp',2)]:
 with (ROOT/f'reports/smoke_{model}.log').open('w') as log:
  p=subprocess.Popen([PYTHON,str(ROOT/'scripts/train_model.py'),'--model',model,'--smoke'],env=env(gpu),stdout=log,stderr=subprocess.STDOUT)
  save('smoke',model=model,pid=p.pid,gpu=gpu)
  rc=p.wait()
 if rc:
  save('smoke_failed',model=model,returncode=rc,log=str(ROOT/f'reports/smoke_{model}.log'));sys.exit(rc)
 print('SMOKE PASSED',model,flush=True)
jobs={}
for model,gpu in [('dense',0),('bsp',2)]:
 out=ROOT/'outputs'/f'microwave_{model}_seed42';out.mkdir(exist_ok=True,parents=True)
 with (out/'console.log').open('a') as log:
  p=subprocess.Popen([PYTHON,str(ROOT/'scripts/train_model.py'),'--model',model],env=env(gpu),stdout=log,stderr=subprocess.STDOUT)
 jobs[model]=(p,gpu)
 print('TRAIN LAUNCHED',model,'GPU',gpu,'PID',p.pid,flush=True)
while True:
 status={model:dict(pid=p.pid,gpu=gpu,returncode=p.poll(),output=str(ROOT/'outputs'/f'microwave_{model}_seed42')) for model,(p,gpu) in jobs.items()}
 done=all(v['returncode'] is not None for v in status.values())
 save('finished' if done else 'training',jobs=status)
 if done:sys.exit(0 if all(v['returncode']==0 for v in status.values()) else 1)
 time.sleep(15)
