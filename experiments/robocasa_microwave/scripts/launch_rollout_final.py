from pathlib import Path
import os,sys,json,time,subprocess,fcntl
ROOT=Path(__file__).resolve().parents[1]
PYTHON=os.environ.get('EVAL_PYTHON',sys.executable)
BASE=ROOT/'outputs/rollout_final'
STATE=ROOT/'reports/rollout_final_status.json'
lock=(ROOT/'reports/rollout_final.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
reference=json.loads((ROOT/'reports/rollout_interim_comparison.json').read_text())
seeds=reference['protocol']['seeds'];assert len(seeds)==50 and len(set(seeds))==50
def save(stage,**extra):
 p=STATE.with_suffix('.tmp');p.write_text(json.dumps({'stage':stage,'time':time.time(),'launcher_pid':os.getpid(),'checkpoint_epoch':600,'training_epochs':601,'n_test_each':50,**extra},indent=2));p.replace(STATE)
def launch(model,shard,selected):
 gpu='0' if model=='dense' else '2'
 e=os.environ.copy();e.update(PYTHONPATH=os.environ.get('MICROWAVE_EVAL_DEPS','/tmp/bsp-microwave-eval-deps')+os.pathsep+os.environ.get('MICROWAVE_DEPS','/tmp/bsp-microwave-deps')+os.pathsep+e.get('PYTHONPATH',''),CUDA_VISIBLE_DEVICES=gpu,MUJOCO_EGL_DEVICE_ID=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1')
 out=BASE/model/f'shard_{shard}'
 with (ROOT/f'reports/rollout_final_{model}_{shard}.log').open('a') as f:
  return subprocess.Popen([PYTHON,str(ROOT/'scripts/rollout_mid.py'),'--model',model,'--checkpoint',str(BASE/f'checkpoints/{model}.ckpt'),'--output-dir',str(out),'--seeds',','.join(map(str,selected)),'--video-count','2' if shard==0 else '0'],env=e,stdout=f,stderr=subprocess.STDOUT)
jobs={}
for m in ['dense','bsp']:
 for shard,(offset,count) in enumerate([(0,13),(13,13),(26,12),(38,12)]):jobs[(m,shard)]=launch(m,shard,seeds[offset:offset+count])
while True:
 progress={}
 for m in ['dense','bsp']:
  records=[json.loads(p.read_text()) for shard in (BASE/m).glob('shard_*') for p in shard.glob('episode_*.json')]
  good=[x for x in records if x['status']=='completed']
  progress[m]={'n_test':len(good),'n_failed':len(records)-len(good),'successes':sum(x['success'] for x in good),'success_rate':sum(x['success'] for x in good)/len(good) if good else None,'shards':[{'shard':i,'pid':jobs[(m,i)].pid,'returncode':jobs[(m,i)].poll()} for i in range(4)]}
 done=all(p.poll() is not None for p in jobs.values());save('verifying' if done else 'rollout',jobs=progress)
 if done:break
 time.sleep(15)
try:
 assert all(p.returncode==0 for p in jobs.values()),'evaluation subprocess failed'
 subprocess.run([sys.executable,str(ROOT/'scripts/summarize_rollout.py'),'--run-dir',str(BASE),'--expected-epoch','600','--tag','final'],check=True)
except Exception as exc:
 save('failed',error=repr(exc),jobs=progress);raise
