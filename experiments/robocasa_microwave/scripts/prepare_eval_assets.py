from pathlib import Path
import ast,urllib.request,zipfile,time,json
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'third_party/robocasa_eval/robocasa'
s=ast.parse((base/'scripts/download_kitchen_assets.py').read_text())
items=[]
for node in ast.walk(s):
 if isinstance(node,ast.keyword) and isinstance(node.value,ast.Call):
  fields={k.arg:k.value for k in node.value.keywords}
  if 'url' in fields and 'folder' in fields:
   url=ast.literal_eval(fields['url']);folder=ast.literal_eval(fields['folder'].args[1]);items.append((node.arg,url,base/folder))
cache=ROOT/'data/assets_downloads';cache.mkdir(parents=True,exist_ok=True)
for name,url,dest in items:
 if name=='aigen_objs':continue  # Task default obj_registries is objaverse only.
 done=cache/(name+'.done')
 if done.exists():continue
 target=cache/(name+'.zip')
 print('download',name,url,flush=True)
 if not target.exists():
  for attempt in range(3):
   try:
    with urllib.request.urlopen(url,timeout=90) as response,(target.with_suffix('.partial')).open('wb') as out:
     while True:
      chunk=response.read(4*1024*1024)
      if not chunk:break
      out.write(chunk)
    target.with_suffix('.partial').replace(target);break
   except Exception as e:
    print('retry',name,str(e),flush=True)
    if attempt==2:raise
 print('extract',name,target.stat().st_size,flush=True)
 dest.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(target) as z:
  for entry in z.infolist():
   if '..' in Path(entry.filename).parts or Path(entry.filename).is_absolute():raise RuntimeError('unsafe zip')
  z.extractall(dest.parent)
 done.write_text(str(time.time()))
print('assets ready',flush=True)
