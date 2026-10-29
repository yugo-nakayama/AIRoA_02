import sys; sys.path.insert(0,"fake")
import numpy as np, cv2, joblib
import genesis_sort as G
clf = joblib.load("shape_mlp_3d.joblib")
rng=np.random.default_rng(1)
W,H=G.CAM_RES; f=(H/2)/np.tan(np.deg2rad(G.CAM_FOV)/2); cz=G.CAM_POS[2]
Mtrue=np.array([[0,-1],[-1,0]],float)  # some unknown true convention
Minv=np.linalg.inv(Mtrue)
def world2pix(p,z): d=Minv@(p-np.array(G.CAM_POS[:2])); return d*f/(cz-z)+np.array([W/2,H/2])
objs=[]; depth=np.full((H,W),cz,np.float32)
for c in ["circle","square","triangle"]*2:
    while True:
        p=np.array([rng.uniform(*G.WORK_X),rng.uniform(*G.WORK_Y)])
        if all(np.linalg.norm(p-o[1])>0.1 for o in objs): break
    objs.append((c,p)); yaw=rng.uniform(0,6.28)
    if c=="circle": poly=[p+0.022*np.array([np.cos(t),np.sin(t)]) for t in np.linspace(0,6.28,40)]
    elif c=="square": poly=[p+0.0283*np.array([np.cos(yaw+t),np.sin(yaw+t)]) for t in np.arange(4)*np.pi/2]
    else: poly=[p+0.0289*np.array([np.cos(yaw+t),np.sin(yaw+t)]) for t in np.arange(3)*2*np.pi/3]
    m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[np.array([world2pix(q,0.04) for q in poly]).astype(np.int32)],1)
    depth[m>0]=cz-0.04
depth+=rng.normal(0,0.002,depth.shape).astype(np.float32)
class Cam: pass
def fake_render(cam,depth=False, _d=depth): return (np.zeros((H,W,3),np.uint8),_d) if depth else np.zeros((H,W,3),np.uint8)
G.render=fake_render
tc=G.TopCamera(Cam())
tc.M=np.array([[1,0],[0,1]],float)
_,bl=G.perceive(tc,clf)
err=tc.calibrate(bl,np.array([o[1] for o in objs]))
_,bl=G.perceive(tc,clf)
print("blobs",len(bl),"calib err mm",round(err*1000,2))
for b in bl:
    i=np.argmin([np.linalg.norm(b["pos"][:2]-o[1]) for o in objs])
    print(objs[i][0],"->",b["cls"],round(b["conf"],2),"pos err mm",round(np.linalg.norm(b["pos"][:2]-objs[i][1])*1000,1),"z",round(b["pos"][2],3))
