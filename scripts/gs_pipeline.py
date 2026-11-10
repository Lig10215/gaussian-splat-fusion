"""File-based fusion, material edits and measured support surfaces."""
from pathlib import Path
from contextlib import ExitStack
import numpy as np
from scipy.spatial.transform import Rotation
from gs_core import *
from gs_regions import *
def source_options(s):
 return {k:s[k]for k in ['opacity','scale','quaternion','allow_cast','allow_extras']if k in s}
def resolve(root,p):return (root/Path(p)).resolve()if not Path(p).is_absolute()else Path(p)
def source_matrix(spec,root):
 value=spec.get('transform',np.eye(4))
 if isinstance(value,str):
  value=load_json(resolve(root,value));
  if isinstance(value,dict)and value.get('accepted')is False:raise ValueError('Registration file failed its held-out controls')
  value=value.get('matrix',value)if isinstance(value,dict)else value
 M=np.asarray(value,'f8');similarity(M);return M
def fuse_manifest(manifest_path):
 mp=Path(manifest_path).resolve();root=mp.parent;job=load_json(mp)
 if job.get('schema_version')!=1:raise ValueError('Expected schema_version 1')
 if job.get('example_only'):raise ValueError('Copy the example into a workspace, calibrate it, then remove example_only')
 if job.get('ownership')and'frame'not in job:raise ValueError('Explicit ROI query frame required; do not assume global Z is up')
 specs=job['inputs'];lookup={s['id']:i for i,s in enumerate(specs)}
 if len(lookup)!=len(specs):raise ValueError('Duplicate source ID')
 out=resolve(root,job['output'])
 for suffix in ['.provenance.npz','.report.json']:
  if out.with_suffix(suffix).exists():raise FileExistsError(out.with_suffix(suffix))
 chunk=int(job.get('chunk_size',32768));hashes={};counts=[];covcounts=np.zeros(len(specs),'i8')
 with ExitStack()as stack:
  files=[stack.enter_context(SplatFile(resolve(root,s['path']),**source_options(s)))for s in specs]
  matrices=[source_matrix(s,root)for s in specs];degree=max(f.degree for f in files)
  for i,f in enumerate(files):
   hashes[str(f.path)]=sha256(f.path)
   if specs[i].get('sha256')and specs[i]['sha256']!=hashes[str(f.path)]:raise ValueError('Source version/hash mismatch')
   weight=float(specs[i].get('opacity_weight',1))
   if not 0<=weight<=1:raise ValueError('Opacity weight outside [0,1]')
  def chosen(i,d):
   spec=specs[i]
   if spec.get('selection')=='append':
    if not spec.get('generated_support',False):raise ValueError('Append is reserved for explicitly identified generated support geometry')
    keep=np.ones(len(d),bool)
   else:keep=owners(d[:,:3],job,lookup)==i
   if 'select_region'in spec:keep&=region_mask(d[:,:3],spec['select_region'],job.get('frame'))
   if float(spec.get('opacity_weight',1))==0:keep[:]=False
   return keep
  for i,f in enumerate(files):
   n=0
   for _,d in f.chunks(chunk):n+=int(sum(chosen(i,transform_rows(d,matrices[i]))))
   if n==0 and not specs[i].get('optional',False):raise ValueError(f"Source {specs[i]['id']} contributes zero points; fix ownership/alignment, or explicitly mark optional")
   counts.append(n)
  total=sum(counts)
  if total<=0:raise ValueError('Empty fusion')
  sid=np.empty(total,'i4');rowid=np.empty(total,'i8');cursor=0
  def chunks():
   nonlocal cursor
   for i,f in enumerate(files):
    for start,d in f.chunks(chunk):
     d=transform_rows(d,matrices[i]);keep=chosen(i,d);rows=np.arange(start,start+len(d),dtype='i8')[keep];d=promote(d[keep],degree)
     if not len(d):continue
     if job.get('bound_covariance',False):
      for reg in job.get('ownership',[]):
       if reg['owner']!=specs[i]['id']or'ceiling'not in reg:continue
       hit=region_mask(d[:,:3],reg,job.get('frame'))
       if np.any(hit):part,n=bound_to_ceiling(d[hit],reg,job.get('frame'));d[hit]=part;covcounts[i]+=n
     d,_=attenuate(d,float(specs[i].get('opacity_weight',1)))
     sid[cursor:cursor+len(d)]=i;rowid[cursor:cursor+len(d)]=rows;cursor+=len(d);yield d
   for f in files:
    if sha256(f.path)!=hashes[str(f.path)]:raise ValueError('Input changed while fusing: '+str(f.path))
  up=job.get('up')
  if up is None and job.get('frame'):up=frame(job['frame'])[1][:,2]
  if up is None and files[0].up is not None:up=similarity(matrices[0])[1]@np.asarray(files[0].up)
  report=write_atomic(out,degree,total,chunks(),up,[f.path for f in files])
  if cursor!=total:raise ValueError('Internal provenance count mismatch')
  prov=out.with_suffix('.provenance.npz');np.savez(prov,source_id=sid,source_row=rowid)
  report.update(dict(manifest=str(mp),provenance=str(prov),inputs=[dict(id=s['id'],path=str(f.path),sha256=hashes[str(f.path)],selected=counts[i],transform=matrices[i].tolist(),degree=f.degree,normalization_dropped_fields=f.extras,covariance_bound_operations=int(covcounts[i]))for i,(s,f)in enumerate(zip(specs,files))],ownership=job.get('ownership',[]),frame=job.get('frame'),primary_identity_transform=bool(np.array_equal(matrices[0],np.eye(4))),validation='Binary readback and hashes passed; visual, registration and occlusion acceptance still required'))
 save_json(out.with_suffix('.report.json'),report);return report
def transform_file(path,output,M,options=None):
 M=np.asarray(M,'f8')
 with SplatFile(path,**(options or {}))as f:
  before=sha256(f.path);up=similarity(M)[1]@f.up if f.up is not None else None
  def chunks():
   for _,d in f.chunks():yield transform_rows(d,M)
   if sha256(f.path)!=before:raise ValueError('Input changed')
  return write_atomic(output,f.degree,f.n,chunks(),up,[f.path])
def material_file(path,output,indices,anchor=None,gain=(1,1,1),alpha_exponent=1):
 idx=np.asarray(indices,'i8')
 if idx.ndim!=1 or len(np.unique(idx))!=len(idx)or alpha_exponent<=0:raise ValueError('Unique indices and positive alpha_exponent required')
 gain=np.asarray(gain,'f8')
 if gain.shape!=(3,)or np.any(gain<=0)or not np.isfinite(gain).all():raise ValueError('Invalid linear RGB gain')
 with SplatFile(path)as f:
  if idx.size and (idx.min()<0 or idx.max()>=f.n):raise ValueError('Material index out of range')
  before=sha256(f.path)
  def chunks():
   for start,d in f.chunks():
    local=idx[(idx>=start)&(idx<start+len(d))]-start
    if len(local):
     if anchor is not None:
      rgb=np.clip(colors(d[local],anchor),0,1)*gain;d[local,11:14]=(rgb-.5)/C0;d[local,14:]=0
     else:
      rgb=(.5+C0*d[local,11:14])*gain;d[local,11:14]=(rgb-.5)/C0
      if d.shape[1]>14:d[local,14:]=(d[local,14:].reshape(-1,3,(d.shape[1]-14)//3)*gain[None,:,None]).reshape(len(local),-1)
     if alpha_exponent!=1:
      a=1/(1+np.exp(-np.clip(d[local,6].astype('f8'),-50,50)));a=np.clip(1-(1-a)**alpha_exponent,1e-7,1-1e-7);d[local,6]=np.log(a/(1-a))
    yield d
   if sha256(f.path)!=before:raise ValueError('Input changed')
  report=write_atomic(output,f.degree,f.n,chunks(),f.up,[f.path])
 report.update(dict(changed_records=len(idx),allowed_changed_columns=['opacity','f_dc_*','f_rest_*'],geometry_unchanged=True,source_sha256=before));save_json(Path(output).with_suffix('.material.json'),report);return report
def shell_file(config_path,output):
 cfg=load_json(config_path)
 if cfg.get('example_only'):raise ValueError('Calibrate the example to measured geometry before generating support')
 if not cfg.get('measurement_evidence'):raise ValueError('Support geometry requires explicit measurement_evidence')
 origin,B=frame(cfg.get('frame'));poly=np.asarray(cfg['polygon_xy'],'f8');eq=convex(poly);planes=np.asarray(cfg['planes'],'f8');surface_height(np.zeros((1,3)),planes);step=float(cfg['step']);sigma=float(cfg['normal_sigma']);color=np.asarray(cfg['measured_color'],'f8')
 if step<=0 or sigma<=0 or color.shape!=(3,)or np.any((color<0)|(color>1)):raise ValueError('Positive step/thickness and measured RGB [0,1] required')
 offsets=cfg['layer_offsets'];lo=poly.min(0);hi=poly.max(0);estimate=np.prod(np.ceil((hi-lo)/step)+2)*len(offsets)
 if estimate>cfg.get('max_points',5000000):raise ValueError('Surface grid too large; check scene units and step')
 rotations=[]
 for p in planes:
  N=B@np.r_[-p[:2],1.];N/=np.linalg.norm(N);tx=B[:,0]-N*(N@B[:,0]);tx/=np.linalg.norm(tx);ty=np.cross(N,tx);rotations.append(Rotation.from_matrix(np.stack([tx,ty,N],axis=1)).as_quat()[[3,0,1,2]])
 parts=[]
 for dx,dy,dz in offsets:
  x,y=np.meshgrid(np.arange(lo[0],hi[0],step)+dx,np.arange(lo[1],hi[1],step)+dy);xy=np.c_[x.ravel(),y.ravel()];xy=xy[(xy@eq[:,:2].T+eq[:,2]).max(1)<=0];hs=xy@planes[:,:2].T+planes[:,2];pid=hs.argmin(1);z=hs.min(1)+dz;d=np.zeros((len(xy),14),'<f4');d[:,:3]=np.c_[xy,z]@B.T+origin;d[:,3:6]=np.log([step*1.3,step*1.3,sigma]);d[:,6]=float(cfg.get('opacity_logit',10));d[:,11:14]=(color-.5)/C0
  for k,Q in enumerate(rotations):d[pid==k,7:11]=Q
  parts.append(d)
 report=write_atomic(output,0,sum(len(d)for d in parts),iter(parts),B[:,2],[])
 report.update(dict(generated_support=True,source_geometry=cfg.get('measurement_evidence'),material='Constant measured color; no invented photograph/texture',config=str(Path(config_path).resolve())));save_json(Path(output).with_suffix('.support.json'),report);return report

