"""Read COLMAP binary metadata and read-only SQLite SIFT descriptors."""
from pathlib import Path
import struct,sqlite3
from contextlib import closing
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from gs_core import similarity,save_json
CAM_MODELS={0:('SIMPLE_PINHOLE',3),1:('PINHOLE',4),2:('SIMPLE_RADIAL',4),3:('RADIAL',5),4:('OPENCV',8),5:('OPENCV_FISHEYE',8),6:('FULL_OPENCV',12),7:('FOV',5),8:('SIMPLE_RADIAL_FISHEYE',4),9:('RADIAL_FISHEYE',5),10:('THIN_PRISM_FISHEYE',12)}
def read_cameras(path):
 out={}
 with Path(path).open('rb')as f:
  n=struct.unpack('<Q',f.read(8))[0]
  for _ in range(n):
   cid,mid,w,h=struct.unpack('<iiQQ',f.read(24))
   if mid not in CAM_MODELS:raise ValueError(f'Unknown camera model ID {mid}; update from the exact producer specification')
   name,k=CAM_MODELS[mid];params=np.frombuffer(f.read(8*k),'<f8').copy();out[cid]=dict(model=name,width=w,height=h,params=params)
 return out
def read_images(path):
 out={}
 with Path(path).open('rb')as f:
  n=struct.unpack('<Q',f.read(8))[0]
  for _ in range(n):
   vals=struct.unpack('<i7di',f.read(64));chars=bytearray()
   while True:
    b=f.read(1)
    if not b:raise ValueError('Truncated image name')
    if b==b'\0':break
    chars.extend(b)
    if len(chars)>65536:raise ValueError('Invalid image name')
   count=struct.unpack('<Q',f.read(8))[0];obs=np.frombuffer(f.read(24*count),dtype=[('xy','<f8',(2,)),('id','<i8')]).copy()
   if len(obs)!=count:raise ValueError('Truncated observations')
   out[vals[0]]=dict(name=chars.decode('utf-8'),qvec=np.array(vals[1:5]),tvec=np.array(vals[5:8]),camera_id=vals[8],observations=obs)
 return out
def read_points(path):
 ids=[];xyz=[];rgb=[];tracks=[];st=struct.Struct('<Q3d3BdQ')
 with Path(path).open('rb')as f:
  n=struct.unpack('<Q',f.read(8))[0]
  for _ in range(n):
   v=st.unpack(f.read(st.size));ids.append(v[0]);xyz.append(v[1:4]);rgb.append(v[4:7]);track=np.frombuffer(f.read(8*v[-1]),'<i4').reshape(-1,2).copy()
   if len(track)!=v[-1]:raise ValueError('Truncated track')
   tracks.append(track)
 return np.asarray(ids,'i8'),np.asarray(xyz,'f8'),np.asarray(rgb,'u1'),tracks
def camera_json(sparse,image_name,M=None):
 sparse=Path(sparse);cams=read_cameras(sparse/'cameras.bin');ims=read_images(sparse/'images.bin');matches=[v for v in ims.values()if v['name']==image_name]
 if len(matches)!=1:raise ValueError('Image name missing/ambiguous')
 im=matches[0];cam=cams[im['camera_id']];q=im['qvec'];R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();P=-R.T@im['tvec']
 if M is not None:s,Q,t=similarity(M);P=s*Q@P+t;R=R@Q.T
 params=cam['params'];model=cam['model'];out=dict(model=model,position=P.tolist(),rotation=R.tolist(),width=int(cam['width']),height=int(cam['height']),image=image_name,original_params=params.tolist())
 if model in ['SIMPLE_PINHOLE','SIMPLE_RADIAL','RADIAL']:out.update(fx=float(params[0]),fy=float(params[0]),cx=float(params[1]),cy=float(params[2]))
 elif model in ['PINHOLE','OPENCV','OPENCV_FISHEYE','FULL_OPENCV']:out.update(fx=float(params[0]),fy=float(params[1]),cx=float(params[2]),cy=float(params[3]))
 return out
def project_points(points,cam):
 xyz=np.asarray(points);P=np.asarray(cam['position']);R=np.asarray(cam['rotation']);v=(xyz-P)@R.T;z=v[:,2];xy=v[:,:2]/np.maximum(1e-12,z[:,None]);p=np.asarray(cam.get('original_params',[]));model=cam['model']
 if model in ['SIMPLE_RADIAL','RADIAL']:
  r2=(xy*xy).sum(1);factor=1+p[3]*r2
  if model=='RADIAL':factor+=p[4]*r2*r2
  xy*=factor[:,None]
 elif model=='OPENCV':
  x,y=xy.T;r2=x*x+y*y;rad=1+p[4]*r2+p[5]*r2*r2;xy=np.c_[x*rad+2*p[6]*x*y+p[7]*(r2+2*x*x),y*rad+p[6]*(r2+2*y*y)+2*p[7]*x*y]
 elif model not in ['PINHOLE','SIMPLE_PINHOLE']:raise ValueError('Projection not implemented for this model; use calibrated/dewarped faces')
 return xy*[cam['fx'],cam['fy']]+[cam['cx'],cam['cy']],z
def point_features(sparse,database,every=3):
 sparse=Path(sparse);ids,xyz,rgb,_=read_points(sparse/'points3D.bin');order=np.argsort(ids);ids=ids[order];xyz=xyz[order];rgb=rgb[order];ims=read_images(sparse/'images.bin');acc=np.zeros((len(ids),128),'f4');count=np.zeros(len(ids),'i4')
 uri=Path(database).resolve().as_uri()+'?mode=ro'
 with closing(sqlite3.connect(uri,uri=True))as db:
  for iid in sorted(ims)[::max(1,every)]:
   row=db.execute('SELECT rows,cols,data FROM descriptors WHERE image_id=?',(iid,)).fetchone()
   if row is None:continue
   rows,cols,blob=row;obs=ims[iid]['observations']
   if rows!=len(obs)or cols!=128:raise ValueError('Descriptor/observation indexing mismatch; do not match unrelated databases')
   valid=obs['id']>=0;want=obs['id'][valid];ii=np.searchsorted(ids,want);ok=(ii<len(ids));ii2=ii[ok];ok[ok]=ids[ii2]==want[ok];desc=np.frombuffer(blob,'u1').reshape(rows,cols)[valid][ok]
   np.add.at(acc,ii[ok],desc);np.add.at(count,ii[ok],1)
 valid=count>0;acc=acc[valid]/count[valid,None];acc/=np.maximum(np.linalg.norm(acc,axis=1)[:,None],1e-12)
 return dict(ids=ids[valid],xyz=xyz[valid],rgb=rgb[valid],descriptors=acc.astype('f4'),observations=count[valid])
def match_features(source,target,ratio=.75):
 # Mutual nearest neighbors + ratio test; these are candidates, not verified controls.
 a=source['descriptors'];b=target['descriptors']
 if len(a)<3 or len(b)<3:raise ValueError('Insufficient reconstructed feature points')
 dist,ii=cKDTree(b).query(a,k=2,workers=-1);back=cKDTree(a).query(b,k=1,workers=-1)[1];good=(dist[:,0]<ratio*np.maximum(dist[:,1],1e-12))&(back[ii[:,0]]==np.arange(len(a)))
 return dict(source=source['xyz'][good].tolist(),target=target['xyz'][ii[good,0]].tolist(),ratio=(dist[good,0]/np.maximum(dist[good,1],1e-12)).tolist(),note='Tentative SIFT/SfM matches. Filter dynamic objects and verify spatially distributed controls; not independent ground truth.')


