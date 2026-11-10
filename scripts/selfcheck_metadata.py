"""Independent binary fixtures for the COLMAP metadata interfaces."""
from pathlib import Path
import struct,sqlite3
import numpy as np
from scipy.spatial.transform import Rotation
from colmap_io import *
from gs_core import *
def fixture(root,xyz,desc,R=None,P=None):
 root=Path(root);root.mkdir();R=np.eye(3)if R is None else R;P=np.zeros(3)if P is None else P;Q=Rotation.from_matrix(R).as_quat()[[3,0,1,2]];t=-R@P;n=len(xyz)
 with (root/'cameras.bin').open('wb')as f:f.write(struct.pack('<QiiQQ4d',1,1,1,320,240,200,200,160,120))
 cam=(xyz-P)@R.T;uv=cam[:,:2]/cam[:,2,None]*200+[160,120]
 with (root/'images.bin').open('wb')as f:
  f.write(struct.pack('<Qi7di',1,1,*Q,*t,1));f.write(b'frame.jpg\0');f.write(struct.pack('<Q',n))
  for i,p in enumerate(uv):f.write(struct.pack('<ddq',*p,i+10))
 with (root/'points3D.bin').open('wb')as f:
  f.write(struct.pack('<Q',n))
  for i,p in enumerate(xyz):f.write(struct.pack('<Q3d3BdQii',i+10,*p,100,120,130,.1,1,1,i))
 dbpath=root/'database.db'
 with sqlite3.connect(dbpath)as db:
  db.execute('CREATE TABLE descriptors(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB)');db.execute('INSERT INTO descriptors VALUES(1,?,?,?)',(n,128,desc.tobytes()))
 return dbpath
def run(root):
 rng=np.random.default_rng(93);x=rng.uniform(-.3,.3,(32,3));x[:,2]+=2.;desc=rng.integers(0,256,(32,128),dtype='u1');Q=Rotation.from_euler('z',31,degrees=True).as_matrix();M=np.eye(4);M[:3,:3]=1.8*Q;M[:3,3]=[.5,-.2,.7];y=x@M[:3,:3].T+M[:3,3]
 a=Path(root)/'sfm_a';b=Path(root)/'sfm_b';db1=fixture(a,x,desc);db2=fixture(b,y,desc,R=Q.T,P=M[:3,3]);before=sha256(db1)
 cams=read_cameras(a/'cameras.bin');assert cams[1]['model']=='PINHOLE'
 image=read_images(a/'images.bin')[1];assert len(image['observations'])==32
 ids,xyz,rgb,tracks=read_points(a/'points3D.bin');assert np.array_equal(ids,np.arange(10,42));np.testing.assert_allclose(xyz,x)
 ca=camera_json(a,'frame.jpg');cb=camera_json(a,'frame.jpg',M);u,zz=project_points(x,ca);v,zz2=project_points(y,cb);np.testing.assert_allclose(u,v,atol=1e-11)
 one=point_features(a,db1,1);two=point_features(b,db2,1);pairs=match_features(one,two,.75);assert len(pairs['source'])==32;assert sha256(db1)==before
 np.testing.assert_allclose(np.asarray(pairs['source'])@M[:3,:3].T+M[:3,3],pairs['target'],atol=1e-12)
 return ['COLMAP binary cameras/images/points and track indexing','camera Sim3 pose transport and projective invariance','read-only descriptor database, mutual ratio matches and exact synthetic ties']

