import argparse,json
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);args=p.parse_args()
with h5py.File(args.dataset,'r') as f:
 data=f['data']; names=sorted((x for x in data if x.startswith('demo_')),key=lambda x:int(x.split('_')[-1]))
 assert names,'No demonstrations'
 meta=data.attrs.get('env_args',f.attrs.get('env_args',''))
 if isinstance(meta,bytes):meta=meta.decode()
 if meta:
  env=json.loads(meta) if isinstance(meta,str) else str(meta)
 else:env=None
 summaries=[]
 for name in names:
  d=data[name];a=d['actions'][:]
  assert a.ndim==2 and np.isfinite(a).all(),name
  summaries.append({'demo':name,'length':len(a),'action_dim':a.shape[-1]})
 first=data[names[0]]
 report={'path':str(args.dataset.resolve()),'n_demos':len(names),'env_args':env,'action_dims':sorted(set(x['action_dim'] for x in summaries)),'observations':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in first['obs'].items()},'demonstrations':summaries}
 print(json.dumps(report,indent=2,default=str))
