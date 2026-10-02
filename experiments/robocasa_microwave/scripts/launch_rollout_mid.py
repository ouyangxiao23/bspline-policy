from pathlib import Path
import os,sys,json,time,subprocess,fcntl
ROOT=Path(__file__).resolve().parents[1]
PYTHON=os.environ.get('EVAL_PYTHON',sys.executable)
STATE=ROOT/'reports/rollout_status.json'
lock=(ROOT/'reports/rollout.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(stage,**extra):
 p=STATE.with_suffix('.tmp');p.write_text(json.dumps({'stage':stage,'time':time.time(),'launcher_pid':os.getpid(),**extra},indent=2));p.replace(STATE)
def launch(model,extra,logname):
 e=os.environ.copy();e.update(PYTHONPATH=os.environ.get('MICROWAVE_EVAL_DEPS','/tmp/bsp-microwave-eval-deps')+os.pathsep+os.environ.get('MICROWAVE_DEPS','/tmp/bsp-microwave-deps')+os.pathsep+e.get('PYTHONPATH',''),CUDA_VISIBLE_DEVICES='0' if model=='dense' else '2',MUJOCO_EGL_DEVICE_ID='0' if model=='dense' else '2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1')
 with (ROOT/'reports'/logname).open('w') as f:return subprocess.Popen([PYTHON,str(ROOT/'scripts/rollout_mid.py'),'--model',model,*extra],env=e,stdout=f,stderr=subprocess.STDOUT)
markers=['textures','fixtures','objaverse','generative_textures']
while True:
 pending=[m for m in markers if not (ROOT/'data/assets_downloads'/f'{m}.done').exists()]
 if not pending:break
 save('preparing_assets',pending=pending);time.sleep(15)
for model in ['dense','bsp']:
 p=launch(model,['--smoke'],f'rollout_smoke_{model}.log');save('smoke',model=model,pid=p.pid)
 if p.wait()!=0:save('smoke_failed',model=model);sys.exit(1)
a=json.loads((ROOT/'outputs/rollout_mid/dense/smoke.json').read_text());b=json.loads((ROOT/'outputs/rollout_mid/bsp/smoke.json').read_text())
assert a['initial_state_sha256']==b['initial_state_sha256'], 'paired reset mismatch'
p=launch('dense',['--smoke','--replay-demo','4'],'rollout_replay.log');save('replay_check',pid=p.pid)
if p.wait()!=0:save('replay_failed');sys.exit(1)
r=json.loads((ROOT/'outputs/rollout_mid/dense/replay.json').read_text())
if not r['success']:save('replay_failed',result=r);sys.exit(1)
jobs={}
for m in ['dense','bsp']:
 for shard,(offset,count) in enumerate([(0,13),(13,13),(26,12),(38,12)]):
  out=ROOT/f'outputs/rollout_mid/{m}/shard_{shard}'
  jobs[(m,shard)]=launch(m,['--n-test',str(count),'--start-seed',str(100000+offset),'--output-dir',str(out),'--video-count','2' if shard==0 else '0'],f'rollout_{m}_{shard}.log')
while True:
 progress={}
 for m in ['dense','bsp']:
  records=[];reports=[]
  for shard in range(4):
   q=ROOT/f'outputs/rollout_mid/{m}/shard_{shard}/eval_log.json'
   if q.exists():
    report=json.loads(q.read_text());records+=report['episodes'];reports.append(report)
  good=[x for x in records if x['status']=='completed']
  combined={'model':m,'epoch':reports[0]['epoch'] if reports else None,'n_test':len(good),'n_failed':len(records)-len(good),'successes':sum(x['success'] for x in good),'success_rate':sum(x['success'] for x in good)/len(good) if good else None,'protocol':reports[0]['protocol'] if reports else None,'episodes':sorted(records,key=lambda x:x['seed'])}
  q=ROOT/f'outputs/rollout_mid/{m}/eval_log.json';temp=q.with_suffix('.tmp');temp.write_text(json.dumps(combined,indent=2));temp.replace(q)
  progress[m]={k:v for k,v in combined.items() if k not in ['episodes','protocol']}
 done=all(p.poll() is not None for p in jobs.values());save('finished' if done else 'rollout',jobs=progress)
 if done:break
 time.sleep(15)

# Eligibility is determined before policy execution, with matching exclusions.
for replacement in range(20):
 all_records={}
 for model in ['dense','bsp']:
  all_records[model]=[json.loads(p.read_text()) for shard in (ROOT/f'outputs/rollout_mid/{model}').glob('shard_*') for p in shard.glob('episode_*.json')]
  errors=[x for x in all_records[model] if x['status']!='completed' and 'reset already satisfies success' not in x.get('error','')]
  assert not errors,('unresolved rollout errors',model,errors)
 counts={m:sum(x['status']=='completed' for x in rows) for m,rows in all_records.items()}
 assert counts['dense']==counts['bsp'],counts
 if counts['dense']==50:break
 seed=100050+replacement;shard=4+replacement
 save('replacement_initial_state',seed=seed)
 children=[launch(m,['--n-test','1','--start-seed',str(seed),'--output-dir',str(ROOT/f'outputs/rollout_mid/{m}/shard_{shard}'),'--video-count','0'],f'rollout_{m}_replacement_{seed}.log') for m in ['dense','bsp']]
 assert all(p.wait()==0 for p in children)
else:raise RuntimeError('Could not obtain 50 valid initial states')
subprocess.run([sys.executable,str(ROOT/'scripts/summarize_rollout.py')],check=True)
