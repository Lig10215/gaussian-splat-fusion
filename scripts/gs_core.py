"""Named-field 3DGS PLY I/O and proper Sim(3) transport. CPU only."""
from pathlib import Path
import hashlib,json,os,re,uuid
import numpy as np
from scipy.spatial.transform import Rotation
C0=.28209479177387814
BASE=['x','y','z']+[f'scale_{i}'for i in range(3)]+['opacity']+[f'rot_{i}'for i in range(4)]+[f'f_dc_{i}'for i in range(3)]
TYPES={'float':'f4','float32':'f4','double':'f8','float64':'f8','uchar':'u1','uint8':'u1','char':'i1','int8':'i1','ushort':'u2','uint16':'u2','short':'i2','int16':'i2','int':'i4','int32':'i4','uint':'u4','uint32':'u4'}
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def load_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save_json(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
class SplatFile:
 def __init__(self,path,opacity='logit',scale='log',quaternion='wxyz',allow_cast=False,allow_extras=False):
  self.path=Path(path).resolve();self.opacity=opacity;self.scale=scale;self.quaternion=quaternion
  if opacity not in ['logit','alpha']or scale not in ['log','linear']or quaternion not in ['wxyz','xyzw']:raise ValueError('Unknown activation/quaternion convention')
  with self.path.open('rb')as f:
   lines=[];size=0
   while True:
    line=f.readline();size+=len(line)
    if not line or size>1048576:raise ValueError('Missing/oversized PLY header')
    lines.append(line.decode('utf-8').strip())
    if line.strip()==b'end_header':break
   self.offset=f.tell()
  if lines[0]!='ply':raise ValueError('Not PLY')
  fmt=next((x.split()[1]for x in lines if x.startswith('format ')),None)
  if fmt not in ['binary_little_endian','binary_big_endian']:raise ValueError('Only scalar binary PLY supported; convert ASCII/packed formats explicitly')
  elems=[x for x in lines if x.startswith('element ')]
  if len(elems)!=1 or not elems[0].startswith('element vertex '):raise ValueError('Expected only a vertex element, not a mesh/packed container')
  self.n=int(elems[0].split()[2]);props=[x.split()[1:]for x in lines if x.startswith('property ')]
  if self.n<=0:raise ValueError('Empty model')
  if any(len(x)!=2 or x[0]not in TYPES for x in props):raise ValueError('Unsupported/list property')
  names=[x[1]for x in props]
  if len(set(names))!=len(names):raise ValueError('Duplicate property')
  if not set(BASE)<=set(names):raise ValueError('Missing Gaussian attributes; a colored point cloud is not a trained GS model')
  rest=sorted([int(x[7:])for x in names if re.fullmatch(r'f_rest_\d+',x)])
  if rest!=list(range(len(rest)))or len(rest)not in [0,9,24,45]:raise ValueError('Unsupported/noncontiguous real SH degree (supported: 0-3)')
  self.degree={0:0,9:1,24:2,45:3}[len(rest)];self.names=BASE+[f'f_rest_{i}'for i in rest];self.width=len(self.names)
  extras=set(names)-set(self.names);self.extras=sorted(extras)
  if extras-set(['nx','ny','nz'])and not allow_extras:raise ValueError('Unknown fields would be lost: '+str(sorted(extras))+'; inspect and explicitly allow normalization if appropriate')
  ptypes=dict((n,t)for t,n in props)
  if not allow_cast and any(TYPES[ptypes[n]]!='f4'for n in self.names):raise ValueError('Required GS attributes must be float32; explicit allow_cast needed')
  endian='<'if fmt=='binary_little_endian'else'>'
  self.dtype=np.dtype([(n,endian+TYPES[t])for t,n in props]);expected=self.offset+self.n*self.dtype.itemsize
  if expected!=self.path.stat().st_size:raise ValueError(f'PLY byte count mismatch: expected {expected}, got {self.path.stat().st_size}')
  self.raw=np.memmap(self.path,self.dtype,'r',offset=self.offset,shape=(self.n,));self.format=fmt;self.up=None
  for x in lines:
   if x.startswith('comment Vertical axis:'):
    v=np.fromstring(x.split(':',1)[1],sep=' ')
    if v.shape==(3,)and np.linalg.norm(v)>0:self.up=(v/np.linalg.norm(v)).tolist()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 def close(self):
  if getattr(self,'raw',None)is not None:self.raw._mmap.close();self.raw=None
 def rows(self,start=0,end=None):
  d=np.column_stack([self.raw[n][start:end]for n in self.names]).astype('<f4')
  if self.opacity=='alpha':
   a=d[:,6].astype('f8')
   if np.any((a<0)|(a>1)):raise ValueError('Alpha outside [0,1]')
   a=np.clip(a,1e-7,1-1e-7);d[:,6]=np.log(a/(1-a))
  if self.scale=='linear':
   if np.any(d[:,3:6]<=0):raise ValueError('Nonpositive linear scale')
   d[:,3:6]=np.log(d[:,3:6])
  if self.quaternion=='xyzw':d[:,7:11]=d[:,[10,7,8,9]]
  return d
 def chunks(self,size=32768):
  for start in range(0,self.n,size):yield start,self.rows(start,min(start+size,self.n))
 def info(self):
  x=self.rows(0,min(self.n,10000));return dict(path=str(self.path),points=self.n,degree=self.degree,bytes=self.path.stat().st_size,format=self.format,properties=self.dtype.names,ignored_fields=self.extras,up=self.up,activations=dict(opacity=self.opacity,scale=self.scale,quaternion=self.quaternion),sample_bounds=np.percentile(x[:,:3],[0,50,100],axis=0).tolist())
def promote(d,degree):
 old=(d.shape[1]-14)//3;new=(degree+1)**2-1
 if old==new:return d
 if old>new:raise ValueError('Refuse implicit SH truncation')
 out=np.zeros((len(d),14+3*new),'<f4');out[:,:14]=d[:,:14]
 if old:out[:,14:].reshape(-1,3,new)[:,:,:old]=d[:,14:].reshape(-1,3,old)
 return out
def sh_basis(v,degree=3):
 x,y,z=v.T;cols=[-.4886025119029199*y,.4886025119029199*z,-.4886025119029199*x]
 if degree>=2:cols += [1.0925484305920792*x*y,-1.0925484305920792*y*z,.31539156525252005*(3*z*z-1),-1.0925484305920792*x*z,.5462742152960396*(x*x-y*y)]
 if degree>=3:cols += [-.5900435899266435*y*(3*x*x-y*y),2.890611442640554*x*y*z,-.4570457994644658*y*(5*z*z-1),.3731763325901154*z*(5*z*z-3),-.4570457994644658*x*(5*z*z-1),1.445305721320277*z*(x*x-y*y),-.5900435899266435*x*(x*x-3*y*y)]
 return np.column_stack(cols)[:,:((degree+1)**2-1)]if degree else np.empty((len(v),0))
def sh_rotation(R,degree):
 k=(degree+1)**2-1
 if not k:return np.empty((0,0))
 i=np.arange(96)+.5;z=1-2*i/len(i);phi=np.pi*(3-np.sqrt(5))*i;r=np.sqrt(1-z*z);v=np.c_[r*np.cos(phi),r*np.sin(phi),z];X=sh_basis(v,degree);Y=sh_basis(v@R,degree);T=np.zeros((k,k))
 for lo,hi in [(0,3),(3,8),(8,15)]:
  if hi<=k:T[lo:hi,lo:hi]=np.linalg.lstsq(X[:,lo:hi],Y[:,lo:hi],rcond=None)[0].T
 if not np.allclose(X@T.T,Y,atol=3e-11):raise ValueError('SH rotation residual too high')
 return T
def similarity(M):
 M=np.asarray(M,'f8')
 if M.shape!=(4,4)or not np.isfinite(M).all()or not np.allclose(M[3],[0,0,0,1]):raise ValueError('Invalid affine matrix')
 scale=np.cbrt(np.linalg.det(M[:3,:3]))
 if scale<=0:raise ValueError('Reflection/nonpositive scale unsupported; resolve handedness explicitly')
 R=M[:3,:3]/scale
 if not np.allclose(R.T@R,np.eye(3),atol=2e-6):raise ValueError('Not a uniform-scale rigid transform')
 u,_,v=np.linalg.svd(R);R=u@v
 return scale,R,M[:3,3]
def transform_rows(d,M):
 M=np.asarray(M,'f8')
 validate_rows(d)
 if np.array_equal(M,np.eye(4)):return d.copy()
 scale,R,t=similarity(M);out=d.copy();out[:,:3]=d[:,:3]@M[:3,:3].T+t;out[:,3:6]+=np.log(scale)
 old=Rotation.from_quat(d[:,[8,9,10,7]]);rot=Rotation.from_matrix(R)*old;out[:,7:11]=rot.as_quat()[:,[3,0,1,2]]
 degree=int(round(np.sqrt((d.shape[1]-14)/3+1)-1));k=(degree+1)**2-1
 if k:out[:,14:]=(d[:,14:].reshape(-1,3,k)@sh_rotation(R,degree)).reshape(-1,3*k)
 return out
def colors(d,position):
 k=(d.shape[1]-14)//3;degree=int(round(np.sqrt(k+1)-1));rgb=.5+C0*d[:,11:14]
 if k:
  rays=d[:,:3]-np.asarray(position);rays/=np.maximum(np.linalg.norm(rays,axis=1)[:,None],1e-12);rgb=rgb+np.einsum('nct,nt->nc',d[:,14:].reshape(-1,3,k),sh_basis(rays,degree))
 return rgb
def attenuate(d,weights):
 w=np.asarray(weights,'f8')
 if w.ndim==0:w=np.full(len(d),float(w))
 if w.shape!=(len(d),)or not np.isfinite(w).all()or np.any((w<0)|(w>1)):raise ValueError('Weights must be in [0,1]')
 keep=w>0;out=d[keep].copy();w=w[keep];change=w<1
 out[change,6]=np.log(w[change])-np.logaddexp(-out[change,6].astype('f8'),np.log1p(-w[change]))
 return out,keep
def validate_rows(d):
 if not np.isfinite(d).all():raise ValueError('Nonfinite Gaussian data')
 q=np.linalg.norm(d[:,7:11],axis=1)
 if np.any(abs(q-1)>.002):raise ValueError('Nonunit/zero quaternion; diagnose export convention')
 if np.any(abs(d[:,3:6])>60):raise ValueError('Implausible scale encoding; confirm linear/log convention and units')
def write_atomic(path,degree,count,chunks,up=None,source_paths=()):
 dst=Path(path).resolve();dst.parent.mkdir(parents=True,exist_ok=True)
 if dst in {Path(p).resolve()for p in source_paths}:raise ValueError('Source overwrite forbidden')
 if dst.exists():raise FileExistsError(dst)
 names=BASE+[f'f_rest_{i}'for i in range(3*((degree+1)**2-1))];lines=['ply','format binary_little_endian 1.0','comment Gaussian splat fusion; wxyz quaternions; log-scales; logit opacity']
 if up is not None:
  u=np.asarray(up,'f8')
  if u.shape!=(3,)or not np.isfinite(u).all()or np.linalg.norm(u)<1e-12:raise ValueError('Invalid up vector')
  lines+=['comment Vertical axis: '+' '.join(f'{v:.12g}'for v in u/np.linalg.norm(u))]
 lines += [f'element vertex {count}']+[f'property float {n}'for n in names]+['end_header']
 header=('\n'.join(lines)+'\n').encode('ascii');tmp=dst.with_name(dst.name+'.'+uuid.uuid4().hex+'.partial');digest=hashlib.sha256();seen=0
 try:
  with tmp.open('xb')as f:
   f.write(header)
   for d in chunks:
    if d.shape[1]!=len(names):raise ValueError('Wrong output SH width')
    validate_rows(d);payload=np.asarray(d,dtype='<f4',order='C').tobytes();f.write(payload);digest.update(payload);seen+=len(d)
   f.flush();os.fsync(f.fileno())
  if seen!=count:raise ValueError(f'Output row count mismatch: {seen} != {count}')
  got=hashlib.sha256()
  # Context exit closes the mapping before Windows rename.
  with SplatFile(tmp)as sf:
   for _,d in sf.chunks():validate_rows(d);got.update(d.tobytes())
  if got.digest()!=digest.digest():raise ValueError('Readback payload differs')
  if dst.exists():raise FileExistsError(dst)
  os.rename(tmp,dst)
 finally:
  if tmp.exists():tmp.unlink()
 return dict(path=str(dst),points=count,degree=degree,bytes=dst.stat().st_size,sha256=sha256(dst),readback_verified=True)
def umeyama(x,y,estimate_scale=True):
 x=np.asarray(x,'f8');y=np.asarray(y,'f8')
 if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=3 or len(x)<3:raise ValueError('Need >=3 corresponding 3D pairs')
 a=x-x.mean(0);b=y-y.mean(0)
 if np.linalg.matrix_rank(a,tol=max(np.linalg.norm(a),1)*1e-10)<2:raise ValueError('Collinear ties')
 u,s,vt=np.linalg.svd(b.T@a);parity=np.linalg.det(u@vt);u[:,-1]*=parity;R=u@vt;scale=(s[0]+s[1]+parity*s[2])/(a*a).sum()if estimate_scale else 1.;M=np.eye(4);M[:3,:3]=scale*R;M[:3,3]=y.mean(0)-scale*R@x.mean(0);similarity(M);return M
def fit_pairs(x,y,threshold,seed=17,iterations=2000,holdout_indices=None):
 x=np.asarray(x,'f8');y=np.asarray(y,'f8')
 if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=3 or len(x)<3 or not np.isfinite(x).all()or not np.isfinite(y).all():raise ValueError('Need >=3 finite corresponding 3D pairs')
 if threshold<=0:raise ValueError('Positive target-unit threshold required')
 rng=np.random.default_rng(seed);order=rng.permutation(len(x));nt=max(3,int(len(x)*.2))if len(x)>=10 else 0;test=order[:nt];train=order[nt:]
 
 if holdout_indices is not None:
  test=np.asarray(holdout_indices,'i8')
  if len(np.unique(test))!=len(test)or np.any(test<0)or np.any(test>=len(x)):raise ValueError('Invalid holdout indices')
  train=np.setdiff1d(order,test)
  if len(train)<3:raise ValueError('Fewer than three training ties')
 best=None;score=-1
 for _ in range(iterations):
  try:
   pick=rng.choice(train,3,False);M=umeyama(x[pick],y[pick])
  except (ValueError,np.linalg.LinAlgError):continue
  err=np.linalg.norm(x[train]@M[:3,:3].T+M[:3,3]-y[train],axis=1);inside=err<threshold
  if inside.sum()>score:best=M;score=int(inside.sum())
 if best is None or score<3:raise ValueError('No supported similarity fit')
 for _ in range(3):
  err=np.linalg.norm(x[train]@best[:3,:3].T+best[:3,3]-y[train],axis=1);good=train[err<threshold];best=umeyama(x[good],y[good])
 err=np.linalg.norm(x@best[:3,:3].T+best[:3,3]-y,axis=1)
 def stats(a):return dict(n=len(a),p50=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()))if len(a)else None
 return best,dict(threshold=threshold,inliers=int(sum(err<threshold)),total=len(x),fit=stats(err[good]),holdout=stats(err[test]),holdout_indices=test.tolist(),holdout_kind='caller-supplied'if holdout_indices is not None else 'random tentative matches, not independent ground truth',errors=err.tolist())





