#!/usr/bin/env python3
"""Behavioral checks in an isolated temporary folder; no user model mutations."""
from pathlib import Path
import tempfile,json,struct,subprocess,sys
import numpy as np
from scipy.spatial.transform import Rotation
from gs_core import *
from gs_pipeline import *
from gs_render import *
from gs_verify import verify_preservation
from gs_audit import run_audit
def cloud(n=40,degree=3,seed=7):
 rng=np.random.default_rng(seed);d=np.zeros((n,14+3*((degree+1)**2-1)),'f4');d[:,:3]=rng.uniform(-.4,.4,(n,3));d[:,3:6]=np.log(rng.uniform(.005,.025,(n,3)));d[:,6]=rng.uniform(-1,3,n);d[:,7:11]=Rotation.random(n,random_state=rng).as_quat()[:,[3,0,1,2]];d[:,11:14]=rng.uniform(-.3,.3,(n,3));d[:,14:]=rng.uniform(-.01,.01,(n,d.shape[1]-14));return d
def save(path,d):
 degree=int(round(np.sqrt((d.shape[1]-14)/3+1)-1));return write_atomic(path,degree,len(d),[d],[0,0,1])
def expect_error(fn):
 try:fn()
 except (ValueError,FileExistsError):return
 raise AssertionError('Invalid operation unexpectedly accepted')
def main():
 checks=[]
 with tempfile.TemporaryDirectory(prefix='gaussian_fusion_check_')as tmp:
  root=Path(tmp).resolve();assert root.name.startswith('gaussian_fusion_check_')
  d=cloud();src=root/'a.ply';save(src,d)
  with SplatFile(src)as f:assert np.array_equal(f.rows(),d)
  expect_error(lambda:save(src,d));expect_error(lambda:write_atomic(src,3,len(d),[d],source_paths=[src]))
  broken=root/'broken.ply';broken.write_bytes(src.read_bytes()[:-4]);expect_error(lambda:SplatFile(broken))
  bad=d.copy();bad[0,7:11]=0;expect_error(lambda:save(root/'bad.ply',bad))
  # Reordered standard fields and big endian must preserve the actual attributes.
  names=['x','y','z','nx','ny','nz']+[f'f_dc_{i}'for i in range(3)]+[f'f_rest_{i}'for i in range(45)]+['opacity']+[f'scale_{i}'for i in range(3)]+[f'rot_{i}'for i in range(4)]
  table=np.column_stack([d[:,BASE.index(n)]if n in BASE else d[:,14+int(n[7:])]if n.startswith('f_rest_')else np.zeros(len(d),'f4')for n in names])
  header='\n'.join(['ply','format binary_big_endian 1.0',f'element vertex {len(d)}']+[f'property float {n}'for n in names]+['end_header','']).encode()
  foreign=root/'foreign.ply';foreign.write_bytes(header+table.astype('>f4').tobytes())
  with SplatFile(foreign)as f:assert np.array_equal(f.rows(),d)
  checks.append('binary count, schema reorder, endian, no-overwrite and invalid quaternion')
  R=Rotation.from_euler('xyz',[37,-19,82],degrees=True).as_matrix();M=np.eye(4);M[:3,:3]=2.3*R;M[:3,3]=[1.2,-.6,3.7];new=transform_rows(d,M);P=np.array([1.,-2.,3.]);Pnew=M[:3,:3]@P+M[:3,3]
  np.testing.assert_allclose(colors(d,P),colors(new,Pnew),atol=3e-6)
  oldax=Rotation.from_quat(d[:,[8,9,10,7]]).as_matrix()*np.exp(d[:,3:6])[:,None,:];newax=Rotation.from_quat(new[:,[8,9,10,7]]).as_matrix()*np.exp(new[:,3:6])[:,None,:]
  expected=2.3**2*R@(oldax@oldax.transpose(0,2,1))@R.T
  np.testing.assert_allclose(newax@newax.transpose(0,2,1),expected,atol=2e-8,rtol=3e-5)
  np.testing.assert_allclose(transform_rows(new,np.linalg.inv(M)),d,atol=4e-6)
  w=.37;weighted,_=attenuate(d,w);np.testing.assert_allclose(1/(1+np.exp(-weighted[:,6])),w/(1+np.exp(-d[:,6])),atol=2e-7);assert np.array_equal(attenuate(d,1)[0],d)
  checks.append('Sim3 covariance transport, SH view invariance, inverse transform and physical alpha weights')
  rng=np.random.default_rng(91);x=rng.normal(size=(100,3));y=x@M[:3,:3].T+M[:3,3];y[:20]=rng.normal(size=(20,3))*10
  fitted,rep=fit_pairs(x,y,.005,iterations=500,holdout_indices=list(range(80,100)));np.testing.assert_allclose(fitted,M,atol=1e-10);assert rep['holdout']['p95']<1e-9
  checks.append('outlier-resistant Sim3 with unused trusted holdout')
  ground=root/'g.ply';g=cloud(40,3,11);g[:,:3]=d[:,:3];save(ground,g)
  job={'schema_version':1,'output':'fused.ply','up':[0,0,1],'inputs':[{'id':'a','path':'a.ply'},{'id':'g','path':'g.ply'}],'default_owner':'a','frame':{'origin':[0,0,0],'basis':np.eye(3).tolist()},'ownership':[{'id':'detail','owner':'g','priority':10,'bbox':[[-.2,-.5,-.5],[.2,.5,.5]]}]};manifest=root/'job.json';save_json(manifest,job);before=sha256(src);report=fuse_manifest(manifest);assert sha256(src)==before
  pv=root/'fused.provenance.npz'
  with np.load(pv)as p:owner=p['source_id'];rows=p['source_row']
  with SplatFile(root/'fused.ply')as f:arr=f.rows()
  assert np.all(abs(arr[owner==0,0])>.2);assert np.all(abs(arr[owner==1,0])<=.2)
  assert np.array_equal(arr[owner==0],d[rows[owner==0]])
  required=np.flatnonzero(abs(d[:,0])>.2);keep=verify_preservation(src,root/'fused.ply',pv,required_rows=required);assert keep['all_preserved']
  drop=dict(job);drop['output']='empty_ground.ply';drop['ownership']=[{'owner':'g','bbox':[[20,20,20],[21,21,21]]}];save_json(root/'badjob.json',drop);expect_error(lambda:fuse_manifest(root/'badjob.json'))
  checks.append('single ownership, canonical base preservation, provenance, required rows and missing detail source rejection')
  idx=np.array([2,5,7]);mat=root/'material.ply';material_file(src,mat,idx,anchor=P,alpha_exponent=2)
  with SplatFile(mat)as f:edited=f.rows()
  untouched=np.ones(len(d),bool);untouched[idx]=False;assert np.array_equal(edited[untouched],d[untouched]);assert np.array_equal(edited[:,:6],d[:,:6]);assert np.array_equal(edited[:,7:11],d[:,7:11]);np.testing.assert_allclose(colors(edited[idx],P),colors(d[idx],P),atol=3e-6)
  checks.append('localized material edit preserves geometry and every unrelated record')
  # Full-scene opacity must fail without a slab, pass with the measured slab.
  b=cloud(1,0);b[0,:3]=[0,0,-.25];b[0,3:6]=np.log([.10,.10,.01]);b[0,7:11]=[1,0,0,0];b[0,6]=8;b[0,11:14]=(np.array([.8,.1,.1])-.5)/C0;save(root/'room.ply',b)
  surf={'id':'floor','polygon_xy':[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]],'planes':[[0,0,0]],'interior_margin':.12,'hidden_below':.05,'hidden_sources':[1]}
  cam={'model':'PINHOLE','position':[0,-.6,1],'target':[0,0,0],'up':[0,0,1],'width':96,'height':96,'f':100,'near':.001}
  np.savez(root/'room.provenance.npz',source_id=np.array([1]),source_row=np.array([0]))
  config={'model':'room.ply','provenance':'room.provenance.npz','surfaces':[surf],'tests':[{'id':'view','surface':'floor','camera':cam,'min_pixels':64,'tolerance':2/255}]};save_json(root/'open_audit.json',config);assert not run_audit(root/'open_audit.json',root/'open_audit')['all_passed']
  sh={'polygon_xy':surf['polygon_xy'],'planes':[[0,0,0]],'step':.025,'normal_sigma':.001,'measured_color':[.4,.4,.4],'measurement_evidence':'synthetic ground-truth z=0 fixture','layer_offsets':[[0,0,0],[.0125,.0125,-.004]]};save_json(root/'shell.json',sh);shell_file(root/'shell.json',root/'slab.ply')
  with SplatFile(root/'slab.ply')as f:slab=f.rows()
  sealed=np.vstack([slab,b]);save(root/'sealed.ply',sealed);np.savez(root/'sealed.provenance.npz',source_id=np.r_[np.zeros(len(slab),'i4'),1],source_row=np.r_[np.arange(len(slab)),0]);config['model']='sealed.ply';config['provenance']='sealed.provenance.npz';save_json(root/'closed_audit.json',config);assert run_audit(root/'closed_audit.json',root/'closed_audit')['all_passed']
  pixel=np.zeros((96,96),bool);pixel[44:52,44:52]=True;visible=trace_visibility(b,cam,pixel);hidden=trace_visibility(sealed,cam,pixel,np.r_[np.zeros(len(slab),bool),True]);assert visible.sum()>0;assert hidden.sum()<1e-5
  checks.append('full-scene audit distinguishes an actual floor opening from an opaque slab; contribution trace respects occlusion')
  # Lower SH degrees must be promoted channel-wise, without changing appearance.
  for degree in [0,1,2]:
   low=cloud(8,degree);np.testing.assert_allclose(colors(promote(low,3),P),colors(low,P),atol=1e-7)
  from gs_registration import refine_icp
  known=transform_rows(d,M);save(root/'target.ply',known);initial=M.copy();initial[:3,3]+=[.001,-.002,.0015]
  refined,ir=refine_icp(src,root/'target.ply',initial,.001,.04,iterations=12);np.testing.assert_allclose(refined,M,atol=3e-6)
  from gs_regions import bound_to_ceiling
  flat=cloud(2,0);flat[:,:3]=[[0,0,-.1],[.1,0,-.2]];flat[:,3:6]=np.log([.2,.2,.3]);flat[:,7:11]=[1,0,0,0]
  bounded,n=bound_to_ceiling(flat,{'ceiling':{'planes':[[0,0,0]],'margin':.02}})
  np.testing.assert_array_equal(bounded[:,:3],flat[:,:3]);np.testing.assert_array_equal(bounded[:,11:],flat[:,11:]);basis=Rotation.from_quat(bounded[:,[8,9,10,7]]).as_matrix()*np.exp(bounded[:,3:6])[:,None,:];assert np.all(np.linalg.norm(basis[:,2,:],axis=1)<=(-flat[:,2]-.02)/4.8+1e-6)
  checks.append('mixed-degree channel packing, initialized scale-locked ICP and covariance-only roof bounds')
  from selfcheck_metadata import run as metadata_checks
  checks.extend(metadata_checks(root))
  config['tests'][0]['mode']='probe';config['surfaces'][0]['probe_step']=.04;config['surfaces'][0]['probe_depth']=.2;save_json(root/'probe.json',config);assert run_audit(root/'probe.json',root/'probe')['all_passed']
  checks.append('full-scene opaque-surface probe mode')
  # The same opaque wall/slab must remain opaque in a non-Z scene frame.
  rigid=np.eye(4);rigid[:3,:3]=R;rigid[:3,3]=[.8,-1.1,.6]
  save(root/'rotated.ply',transform_rows(sealed,rigid));rot=json.loads(json.dumps(config));rot['model']='rotated.ply';rot.pop('provenance');rot['surfaces'][0]['frame']={'origin':rigid[:3,3].tolist(),'basis':R.tolist()}
  rc=rot['tests'][0]['camera']
  for field in ['position','target']:rc[field]=(R@np.asarray(cam[field])+rigid[:3,3]).tolist()
  rc['up']=(R@np.asarray(cam['up'])).tolist();save_json(root/'rotated.json',rot);assert run_audit(root/'rotated.json',root/'rotated_audit')['all_passed']
  # Illustrative geometry must not silently become a real repair.
  save_json(root/'example.json',{'example_only':True});expect_error(lambda:shell_file(root/'example.json',root/'example.ply'));expect_error(lambda:run_audit(root/'example.json',root/'example_audit'));expect_error(lambda:camera({'example_only':True}))
  example_job=dict(job);example_job['example_only']=True;save_json(root/'example_job.json',example_job);expect_error(lambda:fuse_manifest(root/'example_job.json'))
  checks.append('rotated scene-frame occlusion and rejection of uncalibrated examples')
  # Exercise the public CLI rather than only importing functions.
  cli=Path(__file__).parent/'gs_cli.py';r=subprocess.run([sys.executable,str(cli),'validate',str(root/'fused.ply')],capture_output=True,text=True);assert r.returncode==0,r.stderr;assert json.loads(r.stdout)['valid']
  checks.append('CLI smoke check')
 print(json.dumps({'passed':True,'checks':checks},indent=2))
if __name__=='__main__':main()


