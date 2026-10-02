from pathlib import Path
import json,h5py
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/turn_off_microwave_human_im.hdf5'
target=ROOT/'data/turn_off_microwave_human_im_indexed.hdf5'
with h5py.File(source,'r') as src,h5py.File(target.with_suffix('.partial'),'w') as dst:
 for k,v in src.attrs.items():dst.attrs[k]=v
 data=dst.create_group('data')
 for k,v in src['data'].attrs.items():data.attrs[k]=v
 names=sorted(src['data'],key=lambda x:int(x.split('_')[-1]))
 mapping={name:f'demo_{i}' for i,name in enumerate(names)}
 for name,new in mapping.items():src.copy(src['data'][name],data,name=new)
 masks=dst.create_group('mask');missing={}
 for key,values in src['mask'].items():
  ids=[x.decode() for x in values[:]]
  missing[key]=[x for x in ids if x not in mapping]
  masks.create_dataset(key,data=[mapping[x].encode() for x in ids if x in mapping])
target.with_suffix('.partial').replace(target)
(ROOT/'reports/dataset_reindex.json').write_text(json.dumps({'source':str(source),'target':str(target),'mapping':mapping,'missing_mask_references':{k:v for k,v in missing.items() if v},'changed':'demo names only; all 54 trajectories, images, actions, states and metadata preserved'},indent=2))
print(target)
