"""Initialized ICP refinement in the target frame; never moves the reference model."""
import numpy as np
from scipy.spatial import cKDTree
from gs_core import SplatFile,similarity,umeyama
def selected_points(path,mask_path=None):
 with SplatFile(path)as f:
  pts=[];mask=np.load(mask_path)if mask_path else None
  if mask is not None:
   if mask.dtype!=bool:
    idx=np.asarray(mask,'i8');mask=np.zeros(f.n,bool);mask[idx]=True
   if mask.shape!=(f.n,):raise ValueError('Point mask length mismatch')
  for start,d in f.chunks():
   keep=d[:,6]>-3
   if mask is not None:keep&=mask[start:start+len(d)]
   pts.append(d[keep,:3])
 return np.vstack(pts).astype('f8')
def voxel_first(x,size):
 if size<=0:raise ValueError('Positive target-unit voxel size required')
 _,ix=np.unique(np.floor(x/size).astype('i8'),axis=0,return_index=True);return np.sort(ix)
def refine_icp(source,target,initial,voxel,max_distance,iterations=30,source_mask=None,target_mask=None):
 if max_distance<=0:raise ValueError('Positive target-unit correspondence distance required')
 M=np.asarray(initial,'f8');similarity(M);x=selected_points(source,source_mask);y=selected_points(target,target_mask);x=x[voxel_first(x@M[:3,:3].T+M[:3,3],voxel)];y=y[voxel_first(y,voxel)];tree=cKDTree(y);history=[];best=M.copy();besterr=np.inf
 for it in range(iterations):
  pred=x@M[:3,:3].T+M[:3,3];dist,ii=tree.query(pred,workers=-1);valid=dist<max_distance
  if sum(valid)<10:raise ValueError('ICP has fewer than ten supported correspondences')
  threshold=np.quantile(dist[valid],.7);valid&=dist<=threshold;err=float(np.sqrt(np.mean(dist[valid]**2)));history.append(dict(iteration=it,rmse=err,pairs=int(sum(valid))))
  if err<besterr:besterr=err;best=M.copy()
  step=umeyama(pred[valid],y[ii[valid]],estimate_scale=False);updated=step@M
  if np.max(abs(updated-M))<1e-9:break
  M=updated
 return best,dict(rmse=besterr,history=history,scale_locked=True,warning='Low ICP error does not validate symmetry, fisheye deformation or non-overlap. Check independent controls and named views.')

