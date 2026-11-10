"""CPU full-SH Gaussian rendering and visibility attribution for local QA."""
import numpy as np
import numba
from PIL import Image
from scipy.spatial.transform import Rotation
from gs_core import SplatFile,colors
def camera(spec):
 if spec.get('example_only'):raise ValueError('Calibrate the example camera before rendering')
 if spec.get('model','PINHOLE')not in ['PINHOLE','SIMPLE_PINHOLE']:raise ValueError('Renderer needs an undistorted pinhole camera; do not reinterpret ERP/fisheye as pinhole')
 P=np.asarray(spec['position'],'f8');w=int(spec['width']);h=int(spec['height'])
 if 'rotation'in spec:R=np.asarray(spec['rotation'],'f8')
 else:
  up=np.asarray(spec['up'],'f8');z=np.asarray(spec['target'],'f8')-P;z/=np.linalg.norm(z);x=np.cross(z,up)
  if np.linalg.norm(x)<1e-10:raise ValueError('Camera direction parallel to up')
  x/=np.linalg.norm(x);y=np.cross(z,x);R=np.stack([x,y,z])
 if P.shape!=(3,)or R.shape!=(3,3)or not np.allclose(R@R.T,np.eye(3),atol=1e-6)or np.linalg.det(R)<.999:raise ValueError('Invalid world-to-camera rotation')
 fx=float(spec.get('fx',spec.get('f',0)));fy=float(spec.get('fy',fx));cx=float(spec.get('cx',w/2));cy=float(spec.get('cy',h/2))
 if min(w,h,fx,fy)<=0:raise ValueError('Positive image size and focal length required')
 return P,R,fx,fy,cx,cy,w,h
def load_model(path):
 with SplatFile(path)as f:return f.rows()
def projected(data,spec):
 P,R,fx,fy,cx,cy,w,h=camera(spec);near=float(spec.get('near',1e-4));cam=(data[:,:3]-P)@R.T;depth=cam[:,2];z=np.maximum(depth,near);uv=cam[:,:2]/z[:,None]*[fx,fy]+[cx,cy];keep=(depth>near)&(uv[:,0]>-.25*w)&(uv[:,0]<1.25*w)&(uv[:,1]>-.25*h)&(uv[:,1]<1.25*h);ids=np.flatnonzero(keep);ids=ids[np.argsort(depth[ids])]
 d=data[ids];cam=cam[ids];z=depth[ids];J=np.zeros((len(d),2,3));J[:,0,0]=fx/z;J[:,1,1]=fy/z;J[:,0,2]=-cam[:,0]*fx/z**2;J[:,1,2]=-cam[:,1]*fy/z**2
 V=(J@R)@Rotation.from_quat(d[:,[8,9,10,7]]).as_matrix()*np.exp(np.clip(d[:,3:6],-30,30))[:,None,:];C=V@V.transpose(0,2,1);aa=C[:,0,0]+.3;bb=C[:,0,1];cc=C[:,1,1]+.3;det=aa*cc-bb*bb;inv=np.c_[cc,-bb,aa]/np.maximum(det[:,None],1e-15);rad=np.minimum(np.sqrt(np.maximum(0,np.c_[aa,cc]))*3.5,[w,h]);alpha=1/(1+np.exp(-np.clip(d[:,6],-50,50)));rgb=np.maximum(0,colors(d,P))
 return ids,uv[ids],inv,rad,alpha,rgb
@numba.njit(cache=True)
def draw(xy,inv,rad,alpha,rgb,w,h,background):
 out=np.empty((h,w,3),np.float32)
 for y in range(h):
  for x in range(w):
   for j in range(3):out[y,x,j]=background[j]
 for i in range(len(xy)-1,-1,-1):
  x,y=xy[i];rx,ry=rad[i];a,b,c=inv[i]
  for v in range(max(0,int(np.floor(y-ry))),min(h,int(np.ceil(y+ry))+1)):
   for u in range(max(0,int(np.floor(x-rx))),min(w,int(np.ceil(x+rx))+1)):
    dx=u-x;dy=v-y;vv=a*dx*dx+2*b*dx*dy+c*dy*dy
    if vv>12:continue
    al=min(.99,alpha[i]*np.exp(-.5*vv))
    if al<1/255:continue
    for j in range(3):out[v,u,j]=out[v,u,j]*(1-al)+rgb[i,j]*al
 return out
def render_array(data,spec):
 _,xy,inv,rad,alpha,rgb=projected(data,spec);w=int(spec['width']);h=int(spec['height']);bg=np.asarray(spec.get('background',[.12,.12,.12]),'f4');return np.clip(draw(xy,inv,rad,alpha,rgb,w,h,bg),0,1)
def render_image(data,spec):return Image.fromarray(np.uint8(render_array(data,spec)*255))
@numba.njit(cache=True)
def trace_kernel(xy,inv,rad,alpha,pixel_mask,point_mask):
 h,w=pixel_mask.shape;T=np.ones((h,w),np.float32);scores=np.zeros(len(xy),np.float64)
 for i in range(len(xy)):
  x,y=xy[i];rx,ry=rad[i];a,b,c=inv[i]
  for v in range(max(0,int(np.floor(y-ry))),min(h,int(np.ceil(y+ry))+1)):
   for u in range(max(0,int(np.floor(x-rx))),min(w,int(np.ceil(x+rx))+1)):
    if not pixel_mask[v,u]:continue
    dx=u-x;dy=v-y;vv=a*dx*dx+2*b*dx*dy+c*dy*dy
    if vv>12:continue
    al=min(.99,alpha[i]*np.exp(-.5*vv))
    if al<1/255:continue
    if point_mask[i]:scores[i]+=T[v,u]*al
    T[v,u]*=1-al
 return scores
def trace_visibility(data,spec,pixel_mask,point_mask=None):
 """Per-row integrated visible alpha in selected pixels. No color heuristics."""
 ids,xy,inv,rad,alpha,rgb=projected(data,spec);pm=np.ones(len(data),bool)if point_mask is None else np.asarray(point_mask,bool)
 mask=np.asarray(pixel_mask,bool)
 if mask.shape!=(int(spec['height']),int(spec['width']))or pm.shape!=(len(data),):raise ValueError('Mask dimensions mismatch')
 scores=trace_kernel(xy,inv,rad,alpha,mask,pm[ids]);out=np.zeros(len(data),'f8');out[ids]=scores;return out
def contact_sheet(images,labels,output,font_path=None):
 from PIL import ImageDraw,ImageFont
 font=ImageFont.truetype(font_path,20)if font_path else ImageFont.load_default();w=max(i.width for i in images);h=max(i.height for i in images);cols=len(images);out=Image.new('RGB',(w*cols,h+32),(28,30,34));dr=ImageDraw.Draw(out)
 for j,(im,label)in enumerate(zip(images,labels)):out.paste(im,(j*w,32));dr.text((j*w+4,4),label,font=font,fill='white')
 out.save(output)

