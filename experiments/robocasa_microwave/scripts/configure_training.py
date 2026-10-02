from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'third_party/bsp/bspline_policy'),str(ROOT/'third_party/bsp/diffusion_policy'),str(ROOT)]
from omegaconf import OmegaConf
import numpy as np
from diffusion_policy.common.sampler import get_val_mask
report=json.loads((ROOT/'reports/dataset_inspection.json').read_text())
assert report['env_args']['env_name']=='TurnOffMicrowave'
obs={}
for key in ['robot0_agentview_left_image','robot0_agentview_right_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','robot0_base_pos','robot0_base_quat','robot0_base_to_eef_pos','robot0_base_to_eef_quat']:
 raw=report['observations'][key]['shape'][1:]
 obs[key]={'shape':[raw[2],raw[0],raw[1]],'type':'rgb'} if 'image' in key else {'shape':raw,'type':'low_dim'}
shape={'obs':obs,'action':{'shape':[report['action_dims'][0]]}}
mask=get_val_mask(report['n_demos'],val_ratio=.1,seed=42)
(ROOT/'reports/split.json').write_text(json.dumps({'seed':42,'val_ratio':.1,'train_demos':np.flatnonzero(~mask).tolist(),'validation_demos':np.flatnonzero(mask).tolist()},indent=2))
base=OmegaConf.load(ROOT/'third_party/bsp/bspline_policy/bspline_policy/config/train_diffusion_unet_real_hybrid_bspline_workspace.yaml')
for name in ['dense','bsp']:
 cfg=OmegaConf.create(OmegaConf.to_container(base,resolve=False))
 for key in ['defaults','hydra','multi_run']:del cfg[key]
 cfg.name='microwave_'+name;cfg.exp_name=name;cfg.task_name='TurnOffMicrowave'
 ds={'_target_':'bspline_policy.dataset.robomimic_replay_bspline_image_dataset.RobomimicReplayBSplineImageDataset' if name=='bsp' else 'diffusion_policy.dataset.robomimic_replay_image_dataset.RobomimicReplayImageDataset','shape_meta':shape,'dataset_path':'${oc.env:MICROWAVE_DATASET}','horizon':16,'pad_before':0 if name=='bsp' else 1,'pad_after':0 if name=='bsp' else 7,'n_obs_steps':2,'abs_action':False,'use_cache':True,'cache_decoded_replay':True,'cache_suffix':'microwave_delta12_128_v1','seed':42,'val_ratio':.1}
 if name=='bsp':ds.update(chunk_size=10,bspline_degree=3,max_error=.002,stride=1,relative_knots=False)
 cfg.task={'name':'TurnOffMicrowave','shape_meta':shape,'dataset':ds,'env_runner':{'_target_':'microwave_support.OfflineRunner'}}
 cfg.policy.crop_shape=[116,116]
 cfg.policy.n_action_steps=16 if name=='bsp' else 8
 if name=='dense':
  cfg.policy._target_='diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy'
  del cfg.policy['bspline_degree']
 cfg.dataloader.batch_size=64;cfg.dataloader.num_workers=4
 cfg.val_dataloader.batch_size=64;cfg.val_dataloader.num_workers=2
 cfg.training.resume=True;cfg.training.num_epochs=601;cfg.training.checkpoint_every=20
 cfg.training.rollout_every=1000000;cfg.training.sample_every=20
 cfg.checkpoint.topk.monitor_key='val_loss';cfg.checkpoint.topk.k=3
 cfg.checkpoint.topk.format_str='epoch={epoch:04d}-val_loss={val_loss:.4f}.ckpt'
 cfg.logging.mode='offline';cfg.logging.name='microwave_'+name+'_seed42';cfg.logging.project='bsp_microwave_reproduction'
 OmegaConf.save(cfg,ROOT/f'configs/train_{name}.yaml')
print('configs and paired trajectory split ready')
