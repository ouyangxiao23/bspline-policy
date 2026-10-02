from pathlib import Path
import json,csv,math,time,argparse
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=ROOT/'outputs/rollout_mid');parser.add_argument('--expected-epoch',type=int,default=380);parser.add_argument('--tag',default='interim');args=parser.parse_args()
BASE=args.run_dir
EPOCH=args.expected_epoch
reports={};valid={};invalid={}
def wilson(k,n):
 z=1.959963984540054;phat=k/n;den=1+z*z/n
 center=(phat+z*z/(2*n))/den
 half=z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/den
 return [center-half,center+half]
for model in ['dense','bsp']:
 records=[]
 for shard in sorted((BASE/model).glob('shard_*')):
  records += [json.loads(p.read_text()) for p in shard.glob('episode_*.json')]
 records.sort(key=lambda x:x['seed'])
 assert len({x['seed'] for x in records})==len(records),'duplicate episode seeds'
 bad=[x for x in records if x['status']!='completed' and 'reset already satisfies success' not in x.get('error','')]
 assert not bad,('unresolved rollout errors',model,bad)
 invalid[model]={x['seed'] for x in records if x['status']!='completed'}
 valid[model]={x['seed']:x for x in records if x['status']=='completed'}
 assert len(valid[model])==50,(model,len(valid[model]))
 assert all(x['epoch']==EPOCH and x['policy_seed']==12345+x['seed']-100000 for x in valid[model].values())
 k=sum(x['success'] for x in valid[model].values())
 reports[model]={'model':model,'epoch':EPOCH,'n_test':50,'n_invalid_initial_states':len(invalid[model]),'n_failed_runtime':0,'successes':k,'success_rate':k/50,'success_rate_wilson_95':wilson(k,50),'episodes':records}
assert valid['dense'].keys()==valid['bsp'].keys()
assert invalid['dense']==invalid['bsp']
assert all(valid['dense'][s]['initial_state_sha256']==valid['bsp'][s]['initial_state_sha256'] for s in valid['dense'])
if args.tag=='final':
 reference=json.loads((ROOT/'reports/rollout_interim_comparison.json').read_text())
 for model in ['dense','bsp']:
  earlier={x['seed']:x for x in reference['results'][model]['episodes'] if x['status']=='completed'}
  assert earlier.keys()==valid[model].keys(), 'final/interim seed mismatch'
  assert all(earlier[s]['initial_state_sha256']==valid[model][s]['initial_state_sha256'] for s in valid[model]), 'final/interim initial state mismatch'
result={'status':'complete','time':time.time(),'task':'TurnOffMicrowave','paper':{'url':'https://arxiv.org/html/2607.09648v1','table':'2(a)','dense_dp_success_rate':.77,'bsp_dp_success_rate':.89},'protocol':{'checkpoint_epoch':EPOCH,'EMA':True,'n_test_each':50,'control_frequency_hz':20,'max_steps':500,'action_execution_cap':8,'paired_initial_states_verified':True,'seeds':sorted(valid['dense']),'excluded_initially_successful_seeds':sorted(invalid['dense']),'source_commits':{'robocasa':'1370b9e0f747d84fb21ed29bacefb1654865301b','robosuite':'2f9bfdce36471db08e31e3c6d918df40f0698ef2'},'mujoco':'3.1.1','paper_protocol_verified':False,'comparison_limits':['Paper specifies 100 Hz spline sampling; this dataset/environment uses 20 Hz control.','Data and environment versions, evaluation count and task-specific checkpoint selection in the paper are not established.','Single training seed; one checkpoint alone cannot establish convergence.']},'results':reports,'paired_outcomes':{'both_success':sum(valid['dense'][s]['success'] and valid['bsp'][s]['success'] for s in valid['dense']),'dense_only':sum(valid['dense'][s]['success'] and not valid['bsp'][s]['success'] for s in valid['dense']),'bsp_only':sum(not valid['dense'][s]['success'] and valid['bsp'][s]['success'] for s in valid['dense'])}}
(ROOT/f'reports/rollout_{args.tag}_comparison.json').write_text(json.dumps(result,indent=2))
with (ROOT/f'reports/rollout_{args.tag}_comparison.csv').open('w') as f:
 w=csv.writer(f,lineterminator="\n");w.writerow(['model','epoch','n_test','successes','success_rate','paper_success_rate','wilson_95_low','wilson_95_high'])
 for m in ['dense','bsp']:
  a=reports[m];w.writerow([m,EPOCH,50,a['successes'],a['success_rate'],result['paper'][m+'_dp_success_rate'],*a['success_rate_wilson_95']])
for m,a in reports.items():(BASE/m/'eval_log.json').write_text(json.dumps(a,indent=2))
(ROOT/('reports/rollout_status.json' if args.tag=='interim' else f'reports/rollout_{args.tag}_status.json')).write_text(json.dumps({'stage':'finished','time':time.time(),'jobs':{m:{k:v for k,v in a.items() if k!='episodes'} for m,a in reports.items()}},indent=2))
print(json.dumps({m:{k:v for k,v in a.items() if k!='episodes'} for m,a in reports.items()},indent=2))
