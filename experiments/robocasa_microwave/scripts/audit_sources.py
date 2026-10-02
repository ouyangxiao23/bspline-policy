from pathlib import Path
import ast,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
spec=json.loads((ROOT/'configs/reproduction.json').read_text())
paths={
 'bsp_policy':ROOT/'third_party/bsp/bspline_policy/bspline_policy/policy/diffusion_unet_bspline_image_policy.py',
 'bsp_representation':ROOT/'third_party/bsp/bspline_policy/bspline_policy/common/bspline_action.py',
 'bsp_training_config':ROOT/'third_party/bsp/bspline_policy/bspline_policy/config/train_diffusion_unet_real_hybrid_bspline_workspace.yaml',
 'microwave_task':ROOT/'third_party/robocasa/robocasa/environments/kitchen/atomic/kitchen_microwave.py',
 'dataset_registry':ROOT/'third_party/robocasa/robocasa/utils/dataset_registry.py'}
checks={}
for name,p in paths.items():
 assert p.is_file(),p
 t=p.read_text()
 if p.suffix=='.py':ast.parse(t)
 checks[name]={'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
task=paths['microwave_task'].read_text()
assert 'class TurnOffMicrowave' in task and 'behavior="turn_off"' in task
assert 'TurnOffMicrowave=dict(' in paths['dataset_registry'].read_text()
assert spec['task']=='TurnOffMicrowave' and spec['representation']['parameter_rows']==16
result={'status':'source_audit_passed','checks':checks,'does_not_verify':['simulator compatibility','dataset availability','BSP rollout correctness','training convergence']}
(ROOT/'reports/source_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
