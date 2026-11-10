"""Full-scene finite-surface occlusion checks, with explicit provenance and camera scope."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.ndimage import binary_erosion
from gs_core import *
from gs_regions import frame,convex,surface_height,region_mask
from gs_render import camera,render_array,load_model
def surface_roi(spec,surface):
 P,R,fx,fy,cx,cy,w,h=camera(spec);origin,B=frame(surface.get('frame'));cp=(P-origin)@B;x,y=np.meshgrid(np.arange(w),np.arange(h));ray=(np.c_[(x.ravel()-cx)/fx,(y.ravel()-cy)/fy,np.ones(w*h)]@R)@B
 eq=convex(surface['polygon_xy']);planes=np.asarray(surface['planes'],'f8');valid=np.zeros(w*h,bool);margin=float(surface.get('interior_margin',0));tol=float(surface.get('intersection_tolerance',1e-6))
 for p in planes:
  den=ray[:,2]-ray[:,:2]@p[:2];distance=(cp[:2]@p[:2]+p[2]-cp[2])/np.where(abs(den)>1e-12,den,1);hit=cp+distance[:,None]*ray;edge=(hit[:,:2]@eq[:,:2].T+eq[:,2]).max(1);height=surface_height(hit,planes)
  valid|=(abs(den)>1e-12)&(distance>0)&(edge<-margin)&(abs(hit[:,2]-height)<tol)&(ray[:,2]<0)
 return valid.reshape(h,w)
def probe_cloud(surface,degree):
 origin,B=frame(surface.get('frame'));poly=np.asarray(surface['polygon_xy'],'f8');eq=convex(poly);step=float(surface['probe_step']);depth=float(surface['probe_depth'])
 if min(step,depth)<=0:raise ValueError('Positive probe_step/probe_depth required')
 lo=poly.min(0);hi=poly.max(0)
 if np.prod(np.ceil((hi-lo)/step)+2)>1000000:raise ValueError('Probe grid too dense; check units')
 x,y=np.meshgrid(np.arange(lo[0],hi[0],step),np.arange(lo[1],hi[1],step));xy=np.c_[x.ravel(),y.ravel()];xy=xy[(xy@eq[:,:2].T+eq[:,2]).max(1)<0];q=np.c_[xy,np.zeros(len(xy))];q[:,2]=surface_height(q,surface['planes'])-depth
 d=np.zeros((len(q),14+3*((degree+1)**2-1)),'f4');d[:,:3]=q@B.T+origin;d[:,3:6]=np.log([step*1.4,step*1.4,step*.1]);d[:,6]=10;d[:,11:14]=(np.array([1,0,1])-.5)/C0
 
 planes=np.asarray(surface['planes'],'f8');hs=q[:,:2]@planes[:,:2].T+planes[:,2];pid=hs.argmin(1)
 for k,p in enumerate(planes):
  N=B@np.r_[-p[:2],1.];norm=np.linalg.norm(N);N/=norm
  if depth/norm<step*.5:raise ValueError('Probe thickness intersects the physical surface; increase probe_depth or reduce probe_step')
  tx=B[:,0]-N*(B[:,0]@N);tx/=np.linalg.norm(tx);ty=np.cross(N,tx);d[pid==k,7:11]=Rotation.from_matrix(np.stack([tx,ty,N],axis=1)).as_quat()[[3,0,1,2]]
 return d
def run_audit(config_path,output_dir):
 cp=Path(config_path).resolve();root=cp.parent;cfg=load_json(cp)
 if cfg.get('example_only'):raise ValueError('Calibrate the example surfaces, sources and cameras before auditing')
 out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);model=(root/cfg['model']).resolve();data=load_model(model);model_hash=sha256(model);degree=int(round(np.sqrt((data.shape[1]-14)/3+1)-1))
 labels=None
 if cfg.get('provenance'):
  with np.load((root/cfg['provenance']).resolve())as p:labels=p['source_id'].copy()
  if labels.shape!=(len(data),):raise ValueError('Provenance length mismatch')
 surfaces={s['id']:s for s in cfg['surfaces']};results=[]
 for test in cfg['tests']:
  surface=surfaces[test['surface']];spec=test['camera'];roi=surface_roi(spec,surface);erosion=int(test.get('mask_erosion_pixels',0))
  if erosion:roi=binary_erosion(roi,iterations=erosion)
  pixels=int(roi.sum());minimum=int(test.get('min_pixels',64))
  if pixels<minimum:raise ValueError(f"Insufficient physical surface coverage for {test['id']}: {pixels}")
  origin,B=frame(surface.get('frame'));q=(data[:,:3]-origin)@B;mode=test.get('mode','recolor');base=render_array(data,spec);selected=0
  if mode=='recolor':
   if surface.get('hidden_indices'):
    ids=np.load((root/surface['hidden_indices']).resolve());hidden=np.zeros(len(data),bool);hidden[ids]=True
   else:
    if labels is None or not surface.get('hidden_sources'):raise ValueError('Recolor requires explicit hidden_sources + provenance or hidden_indices')
    hidden=np.isin(labels,surface['hidden_sources'])
   hidden&=q[:,2]<surface_height(q,surface['planes'])-float(surface['hidden_below'])
   if surface.get('hidden_region'):hidden&=region_mask(data[:,:3],surface['hidden_region'],surface.get('frame'))
   else:
    eq=convex(surface['polygon_xy']);hidden&=(q[:,:2]@eq[:,:2].T+eq[:,2]).max(1)<0
   selected=int(hidden.sum())
   if selected==0:raise ValueError('No hidden content selected; this is not evidence of opacity. Use a probe or correct the source/volume selection.')
   saved=data[hidden,11:].copy();data[hidden,11:14]=(np.array([1,0,1])-.5)/C0;data[hidden,14:]=0
   try:altered=render_array(data,spec)
   finally:data[hidden,11:]=saved
  elif mode=='probe':
   probe=probe_cloud(surface,degree);selected=len(probe);altered=render_array(np.vstack([data,probe]),spec)
  else:raise ValueError('Unknown audit mode')
  delta=abs(base-altered).max(2);v=delta[roi];threshold=float(test.get('tolerance',2/255));changed=int(sum(v>threshold));allowed=int(test.get('allowed_changed_pixels',0))
  row=dict(id=test['id'],surface=surface['id'],mode=mode,selected_hidden_or_probe=selected,tested_pixels=pixels,changed_pixels=changed,max_change_255=float(v.max()*255),mean_change_255=float(v.mean()*255),tolerance_255=threshold*255,allowed_changed_pixels=allowed,passed=changed<=allowed,camera=spec,physical_edge_margin=surface.get('interior_margin',0),mask_erosion_pixels=erosion)
  results.append(row);print(row,flush=True)
  prefix=out/test['id'];Image.fromarray(np.uint8(base*255)).save(str(prefix)+'_base.png');Image.fromarray(np.uint8(altered*255)).save(str(prefix)+'_hidden.png');Image.fromarray(np.uint8(roi)*255).save(str(prefix)+'_roi.png')
 report=dict(model=str(model),model_sha256=model_hash,config=str(cp),config_sha256=sha256(cp),all_passed=all(x['passed']for x in results),scope='Only the listed cameras and selected physical surface interiors; not an all-angle or viewer-equivalence guarantee',results=results)
 if not results:raise ValueError('No audit cases')
 if sha256(model)!=model_hash:raise ValueError('Model changed during audit')
 save_json(out/'audit.json',report);return report

