"""Prepare isolated node dependencies without modifying the shared Python environment."""
from pathlib import Path
import subprocess,urllib.request,tarfile,io,os,sys
ROOT=Path(__file__).resolve().parents[1]
PYTHON=os.environ.get('EVAL_PYTHON',sys.executable)
DEPS=Path(os.environ.get('MICROWAVE_DEPS','/tmp/bsp-microwave-deps'));DEPS.mkdir(exist_ok=True)
subprocess.run([str(PYTHON),'-m','pip','install','--index-url','https://pypi.org/simple','--target',str(DEPS),'--no-deps','robomimic==0.2.0','imagecodecs==2024.12.30'],check=True)
archive=urllib.request.urlopen('https://codeload.github.com/facebookresearch/pytorch3d/tar.gz/refs/tags/v0.7.7',timeout=90).read()
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as t:
 for m in t.getmembers():
  parts=Path(m.name).parts
  if len(parts)<3 or parts[1]!='pytorch3d' or parts[2] not in ['__init__.py','transforms','common'] or not m.isfile():continue
  if '..' in parts:raise RuntimeError('unsafe archive')
  path=DEPS.joinpath(*parts[1:]);path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(t.extractfile(m).read())
print(DEPS)
