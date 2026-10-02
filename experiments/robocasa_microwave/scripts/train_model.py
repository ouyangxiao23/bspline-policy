from pathlib import Path
import sys,argparse,json,time,os
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'third_party/bsp/bspline_policy'),str(ROOT/'third_party/bsp/diffusion_policy'),str(ROOT)]
import torch,hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusion_policy.common.pytorch_util import dict_apply
p=argparse.ArgumentParser();p.add_argument('--model',choices=['dense','bsp'],required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
OmegaConf.register_new_resolver('eval',eval,replace=True)
os.environ.setdefault('MICROWAVE_DATASET',str(ROOT/'data/turn_off_microwave_human_im_indexed.hdf5'))
cfg=OmegaConf.load(ROOT/f'configs/train_{args.model}.yaml');OmegaConf.resolve(cfg)
out=ROOT/'outputs'/('microwave_'+args.model+'_seed42');out.mkdir(parents=True,exist_ok=True)
OmegaConf.save(cfg,out/'resolved_config.yaml')
if args.smoke:
 torch.manual_seed(42)
 ds=hydra.utils.instantiate(cfg.task.dataset)
 normalizer=ds.get_normalizer()
 batch=next(iter(DataLoader(ds,batch_size=int(cfg.dataloader.batch_size),num_workers=0)))
 assert batch['action'].shape[1:]==(16,13 if args.model=='bsp' else 12)
 assert all(torch.isfinite(t).all() for t in batch['obs'].values()) and torch.isfinite(batch['action']).all()
 model=hydra.utils.instantiate(cfg.policy);model.set_normalizer(normalizer);model.cuda()
 batch=dict_apply(batch,lambda x:x.cuda())
 opt=hydra.utils.instantiate(cfg.optimizer,params=model.parameters())
 torch.cuda.reset_peak_memory_stats();start=time.time()
 loss=model.compute_loss(batch);assert torch.isfinite(loss)
 loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 opt.step();opt.zero_grad(set_to_none=True)
 model.eval()
 with torch.no_grad():prediction=model.predict_action(batch['obs'])
 assert torch.isfinite(prediction['action_pred']).all()
 val=ds.get_validation_dataset()
 assert not (ds.train_mask & val.train_mask).any()
 result={'model':args.model,'status':'passed','loss':loss.item(),'samples':len(ds),'validation_samples':len(val),'parameters':sum(p.numel() for p in model.parameters()),'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'seconds':time.time()-start,'action_shape':list(batch['action'].shape),'prediction_shape':list(prediction['action_pred'].shape)}
 (ROOT/f'reports/smoke_{args.model}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
else:
 cls=hydra.utils.get_class(cfg._target_);workspace=cls(cfg,output_dir=str(out));workspace.run()
