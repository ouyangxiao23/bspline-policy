"""Paired interim rollout; independent simulator source, original policy/EMA weights."""
from pathlib import Path
import os,sys,argparse,json,time,random,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'third_party/robocasa_eval'),str(ROOT/'third_party/robosuite_eval'),str(ROOT/'third_party/bsp/bspline_policy'),str(ROOT/'third_party/bsp/diffusion_policy'),str(ROOT)]
import numpy as np,torch,hydra,dill,h5py
from omegaconf import OmegaConf
from scipy.interpolate import BSpline
import robosuite,robocasa,mujoco
assert mujoco.__version__=="3.1.1",mujoco.__version__
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path);p.add_argument('--seeds');p.add_argument('--output-dir',type=Path);p.add_argument('--model',choices=['dense','bsp'],required=True);p.add_argument('--n-test',type=int,default=50);p.add_argument('--start-seed',type=int,default=100000);p.add_argument('--max-steps',type=int,default=500);p.add_argument('--video-count',type=int,default=2);p.add_argument('--smoke',action='store_true');p.add_argument('--replay-demo',type=int);args=p.parse_args()
OUT=args.output_dir or ROOT/'outputs/rollout_mid'/args.model;OUT.mkdir(parents=True,exist_ok=True)
meta=json.loads((ROOT/'reports/dataset_inspection.json').read_text())['env_args']
kwargs=meta['env_kwargs'].copy();kwargs.update(has_renderer=False,has_offscreen_renderer=True,use_camera_obs=True,control_freq=20)
OmegaConf.register_new_resolver('eval',eval,replace=True)
payload=torch.load(args.checkpoint or ROOT/f'outputs/rollout_mid/checkpoints/{args.model}.ckpt',map_location='cpu',weights_only=False)
cfg=payload['cfg'];OmegaConf.resolve(cfg)
policy=hydra.utils.instantiate(cfg.policy);policy.load_state_dict(payload['state_dicts']['ema_model' if cfg.training.use_ema else 'model']);policy.cuda().eval()
epoch=dill.loads(payload['pickles']['epoch']);del payload
assert ('BSpline' in cfg.policy._target_)==(args.model=='bsp'), 'checkpoint policy type mismatch'
print('CHECKPOINT',args.model,'epoch',epoch,'training_epochs',cfg.training.num_epochs,'EMA',cfg.training.use_ema,flush=True)
keys=cfg.shape_meta.obs
# Exact same knot safety rule used by the upstream deployment entrypoint.
def safer_knots(knots):
 knots=np.asarray(knots,dtype=np.float64).copy()
 for idx in range(1,len(knots)):
  if knots[idx]<knots[idx-1]:knots[idx]=knots[idx-1]+1e-6
 return knots
def prepare(obs):
 out={}
 for k,spec in keys.items():
  a=np.asarray(obs[k])
  if spec.get('type')=='rgb':a=np.moveaxis(a[::-1],-1,0).astype(np.float32)/255.
  else:a=a.astype(np.float32)
  assert tuple(a.shape)==tuple(spec['shape']),(k,a.shape,spec['shape'])
  out[k]=a.copy()
 return out
def decode(params):
 knots=safer_knots(params[:,0]); degree=int(cfg.policy.bspline_degree)
 spline=BSpline(knots,params[:-(degree+1),1:],degree)
 # Dataset knot times are frame indices relative to the first observation.
 # Two history observations put the current environment frame at local time 1.
 end=knots[-degree-1]; n=max(1,min(8,int(np.ceil(end-1))))
 assert end>knots[degree], 'invalid spline domain'
 times=np.clip(1+np.arange(n,dtype=np.float64),knots[degree],np.nextafter(end,-np.inf))
 actions=spline(times).astype(np.float32)
 assert np.isfinite(actions).all(),'nonfinite spline actions'
 return actions
records=[]
test_seeds=[int(x) for x in args.seeds.split(',')] if args.seeds else list(range(args.start_seed,args.start_seed+args.n_test))
for i in range(1 if args.smoke else len(test_seeds)):
 seed=test_seeds[i];random.seed(seed);np.random.seed(seed);torch.manual_seed(12345+seed-100000);torch.cuda.manual_seed_all(12345+seed-100000)
 path=OUT/f'episode_{seed}.json'
 if path.exists() and not args.smoke:records.append(json.loads(path.read_text()));continue
 started=time.time();env=None;writer=None
 try:
  env=robosuite.make(meta['env_name'],**dict(kwargs,seed=seed));obs=env.reset()
  print('RESET',args.model,seed,'action_dim',env.action_dim,'control_freq',env.control_freq,flush=True)
  if args.replay_demo is not None:
   with h5py.File(ROOT/'data/turn_off_microwave_human_im_indexed.hdf5','r') as f:
    d=f['data'][f'demo_{args.replay_demo}'];xml=d.attrs['model_file'];states=d['states'][:];acts=d['actions'][:]
    # Stored XML refers to the dataset author's package paths.
    import re
    xml=re.sub(r'[^" ]*/robocasa/robocasa/models/assets/',str(ROOT/'third_party/robocasa_eval/robocasa/models/assets')+'/',xml)
    xml=re.sub(r'[^" ]*/robosuite/robosuite/models/assets/',str(ROOT/'third_party/robosuite_eval/robosuite/models/assets')+'/',xml)
    from robocasa.scripts.playback_dataset import reset_to
    reset_to(env,{'model':xml,'states':states[0],'ep_meta':d.attrs.get('ep_meta')})
    for act in acts:obs,reward,done,info=env.step(act)
    result={'status':'replay_done','success':bool(env._check_success()),'steps':len(acts),'seed':seed,'model':args.model,'epoch':epoch}
   (OUT/'replay.json').write_text(json.dumps(result,indent=2));print(result,flush=True);break
  assert not env._check_success(), 'reset already satisfies success; invalid evaluation initialization'
  initial_state_hash=hashlib.sha256(env.sim.get_state().flatten().tobytes()).hexdigest()
  initial_scene={'layout_id':env.layout_id,'style_id':env.style_id}
  if not args.smoke and i<args.video_count:
   import imageio
   writer=imageio.get_writer(str(OUT/f'episode_{seed}.mp4'),fps=20)
  current=prepare(obs);history=[current,current];policy.reset();success=False;steps=0;invalid=0;clip_count=0;max_action_magnitude=0.
  while steps<(2 if args.smoke else args.max_steps):
   inputs={k:torch.from_numpy(np.stack([h[k] for h in history])).unsqueeze(0).cuda() for k in keys}
   with torch.no_grad():pred=policy.predict_action(inputs)['action'][0].cpu().numpy()
   actions=decode(pred) if args.model=='bsp' else pred
   low,high=env.action_spec
   for act in actions:
    max_action_magnitude=max(max_action_magnitude,float(np.max(np.abs(act))))
    clipped=np.clip(act,low,high);clip_count+=int(np.any(clipped!=act))
    obs,reward,done,info=env.step(clipped);steps+=1
    history=[history[-1],prepare(obs)]
    if writer is not None:writer.append_data(np.concatenate([obs[k][::-1] for k in keys if keys[k].get('type')=='rgb'],axis=1))
    success=bool(env._check_success())
    if success or steps>=(2 if args.smoke else args.max_steps):break
   if success:break
  result={'status':'completed','model':args.model,'epoch':epoch,'seed':seed,'policy_seed':12345+seed-100000,'success':success,'steps':steps,'seconds':time.time()-started,'clipped_steps':clip_count,'max_action_magnitude':max_action_magnitude,'initial_state_sha256':initial_state_hash,'scene':initial_scene}
 except Exception as e:
  import traceback;traceback.print_exc()
  result={'status':'failed','model':args.model,'epoch':epoch,'seed':seed,'error':repr(e),'seconds':time.time()-started}
 finally:
  if writer is not None:writer.close()
  if env is not None:env.close()
 if args.smoke:
  (OUT/'smoke.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
  if result['status']!='completed':sys.exit(1)
 else:
  path.write_text(json.dumps(result,indent=2));records.append(result)
  good=[x for x in records if x['status']=='completed']
  report={'model':args.model,'epoch':epoch,'n_test':len(good),'n_failed':len(records)-len(good),'successes':sum(x['success'] for x in good),'success_rate':sum(x['success'] for x in good)/len(good) if good else None,'protocol':{'environment':'RoboCasa June 2024 snapshot','robosuite':'June 2024 snapshot reports 1.4.1','mujoco':mujoco.__version__,'control_freq':20,'max_steps':args.max_steps,'test_start_seed':args.start_seed,'ema':bool(cfg.training.use_ema),'action_steps_cap':8,'bsp_time_origin':'history first frame; current time 1','bsp_knots':'upstream safer_knots','paper_protocol_verified':False},'episodes':records}
  (OUT/'eval_log.json').write_text(json.dumps(report,indent=2));print(result,flush=True)
