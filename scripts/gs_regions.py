"""Explicit 3D ownership regions. Coordinates remain in the chosen base frame."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
def frame(spec=None):
 spec=spec or {};origin=np.asarray(spec.get('origin',[0,0,0]),'f8');B=np.asarray(spec.get('basis',np.eye(3)),'f8')
 if origin.shape!=(3,)or B.shape!=(3,3)or not np.allclose(B.T@B,np.eye(3),atol=1e-6)or np.linalg.det(B)<.999:raise ValueError('ROI frame needs a proper orthonormal column basis')
 return origin,B
def convex(poly):
 p=np.asarray(poly,'f8')
 if p.ndim!=2 or p.shape[1]!=2 or len(p)<3:raise ValueError('Need a 2D polygon')
 hull=ConvexHull(p)
 if len(hull.vertices)!=len(p):raise ValueError('Polygon must be strictly convex; split concave footprints and holes explicitly')
 return hull.equations
def surface_height(q,planes):
 p=np.asarray(planes,'f8')
 if p.ndim!=2 or p.shape[1]!=3 or len(p)<1:raise ValueError('Planes are z=a*x+b*y+c')
 return np.min(q[:,:2]@p[:,:2].T+p[:,2],axis=1)
def region_mask(xyz,region,frame_spec=None):
 origin,B=frame(region.get('frame',frame_spec));q=(xyz-origin)@B;hit=np.ones(len(q),bool)
 if 'bbox'in region:
  lo,hi=np.asarray(region['bbox'],'f8')
  if lo.shape!=(3,)or np.any(hi<=lo):raise ValueError('Invalid bounding box')
  hit&=np.all((q>=lo)&(q<=hi),axis=1)
 if 'polygon_xy'in region:
  eq=convex(region['polygon_xy']);hit&=(q[:,:2]@eq[:,:2].T+eq[:,2]).max(1)<=0
 if 'z_min'in region:hit&=q[:,2]>=region['z_min']
 if 'z_max'in region:hit&=q[:,2]<=region['z_max']
 if 'ceiling'in region:
  c=region['ceiling'];hit&=q[:,2]<surface_height(q,c['planes'])-float(c.get('margin',0))
 return hit
def owners(xyz,manifest,source_lookup):
 default=manifest.get('default_owner',manifest['inputs'][0]['id'])
 if default not in source_lookup:raise ValueError('Unknown default_owner')
 owner=np.full(len(xyz),source_lookup[default],'i4');priority=np.full(len(xyz),-np.inf)
 for reg in manifest.get('ownership',[]):
  if reg['owner']not in source_lookup:raise ValueError('Unknown owner '+reg['owner'])
  if not ('bbox'in reg or ('polygon_xy'in reg and 'z_min'in reg and ('z_max'in reg or 'ceiling'in reg))):raise ValueError('Ownership needs bounded 3D extent, not an unbounded footprint')
  hit=region_mask(xyz,reg,manifest.get('frame'));p=float(reg.get('priority',0));new=source_lookup[reg['owner']]
  if np.any(hit&(priority==p)&(owner!=new)):raise ValueError('Conflicting equal-priority owners; resolve explicitly')
  use=hit&(p>=priority);owner[use]=new;priority[use]=p
 return owner
def bound_to_ceiling(d,region,frame_spec=None,sigma_factor=4.8):
 """Constrain covariance extent, never move centers or change colors."""
 if 'ceiling'not in region:return d,0
 from gs_core import validate_rows
 validate_rows(d);origin,B=frame(region.get('frame',frame_spec));q=(d[:,:3]-origin)@B;c=region['ceiling'];p=np.asarray(c['planes'],'f8');h=q[:,:2]@p[:,:2].T+p[:,2];pid=h.argmin(1);changed=0;out=d.copy()
 for k,plane in enumerate(p):
  ii=np.flatnonzero(pid==k);N=B@np.r_[-plane[:2],1.];norm=np.linalg.norm(N);N/=norm;dist=(h[ii,k]-q[ii,2])/norm
  clearance=float(c.get('margin',0));limit=np.maximum(float(c.get('minimum_sigma',1e-8)),(dist-clearance)/sigma_factor)
  if np.any(dist<=0):raise ValueError('Center above authoritative roof; crop/refit before covariance bound')
  sig=np.exp(out[ii,3:6]);ax=Rotation.from_quat(out[ii][:,[8,9,10,7]]).as_matrix()
  nstd=np.linalg.norm((N@ax)*sig,axis=1);ratio=np.minimum(1,limit/np.maximum(nstd,1e-15));use=ratio<.999;ii=ii[use];ratio=ratio[use];sig=sig[use];ax=ax[use]
  if not len(ii):continue
  M=np.eye(3)[None]+(ratio-1)[:,None,None]*np.outer(N,N)[None];v=M@(ax*sig[:,None,:]);ev,vec=np.linalg.eigh(v@v.transpose(0,2,1));vec[:,:,0]*=np.where(np.linalg.det(vec)<0,-1,1)[:,None];out[ii,3:6]=.5*np.log(np.maximum(ev,1e-30));out[ii,7:11]=Rotation.from_matrix(vec).as_quat()[:,[3,0,1,2]];changed+=len(ii)
 return out,changed


