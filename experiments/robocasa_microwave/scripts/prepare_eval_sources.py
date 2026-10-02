"""Download independently pinned, old mobile-robot simulator sources."""
from pathlib import Path
import urllib.request,tarfile,io,json
ROOT=Path(__file__).resolve().parents[1]
SOURCES=[('robocasa/robocasa','1370b9e0f747d84fb21ed29bacefb1654865301b','robocasa_eval'),('ARISE-Initiative/robosuite','2f9bfdce36471db08e31e3c6d918df40f0698ef2','robosuite_eval')]
for repo,rev,name in SOURCES:
 dest=ROOT/'third_party'/name
 if (dest/'.source_revision').exists():
  assert (dest/'.source_revision').read_text().strip()==rev
  continue
 data=urllib.request.urlopen(f'https://codeload.github.com/{repo}/tar.gz/{rev}',timeout=120).read()
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
  for member in archive.getmembers():
   parts=Path(member.name).parts[1:]
   if not parts or not member.isfile():continue
   if '..' in parts:raise RuntimeError('unsafe archive')
   path=dest.joinpath(*parts);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.extractfile(member).read())
 (dest/'.source_revision').write_text(rev+'\n')
 print(name,rev,flush=True)
