"""Soft TCP center and translucent local Z guide; display only."""
import cv2
import numpy as np
from robosuite.utils.opencv_renderer import OpenCVRenderer
from robosuite.utils.camera_utils import get_camera_transform_matrix

class EEFAxesRenderer(OpenCVRenderer):
    def __init__(self, previous, site_id, length=.15, show_all=False):
        super().__init__(previous.sim)
        self.width,self.height=previous.width,previous.height
        self.camera_name=previous.camera_name
        self.keypress_callback=previous.keypress_callback
        self.site_id=site_id
        self.axis_length=length

    def frame(self):
        im=self.sim.render(camera_name=self.camera_name,height=self.height,width=self.width)
        im=np.ascontiguousarray(im[::-1,:,::-1])
        origin=np.array(self.sim.data.site_xpos[self.site_id])
        rotation=np.array(self.sim.data.site_xmat[self.site_id]).reshape(3,3)
        points=np.vstack([origin,origin+self.axis_length*rotation[:,2]])
        transform=get_camera_transform_matrix(self.sim,self.camera_name,self.height,self.width)
        projected=(transform@np.c_[points,np.ones(2)].T).T
        valid=np.isfinite(projected).all(axis=1)&(projected[:,2]>1e-5)
        xy=projected[:,:2]/np.maximum(projected[:,2:3],1e-5)
        if valid[0]:
            center=tuple(np.clip(np.round(xy[0]),-10000,10000).astype(int))
            if valid[1]:
                endpoint=tuple(np.clip(np.round(xy[1]),-10000,10000).astype(int))
                overlay=im.copy()
                cv2.line(overlay,center,endpoint,(255,230,130),2,cv2.LINE_AA)
                im=cv2.addWeighted(overlay,.40,im,.60,0)
            if 0<=xy[0,0]<self.width and 0<=xy[0,1]<self.height:
                overlay=im.copy()
                cv2.circle(overlay,center,5,(225,155,225),-1,cv2.LINE_AA)
                im=cv2.addWeighted(overlay,.45,im,.55,0)
        for i,text in enumerate(getattr(self,"hud_lines",[])):
            cv2.rectangle(im,(8,8+27*i),(1000,35+27*i),(25,25,25),-1)
            cv2.putText(im,text,(15,28+27*i),cv2.FONT_HERSHEY_SIMPLEX,.55,(230,230,230),1,cv2.LINE_AA)
        return im

    def render(self):
        cv2.imshow("UR5e - TCP center and soft Z",self.frame())
        key=cv2.waitKey(1)
        if self.keypress_callback:
            self.keypress_callback(key)

def attach_axes(env,length=.15,show_all=False):
    # Attach to the real viewer beneath recording wrappers; only local Z is shown.
    from robosuite.wrappers.wrapper import Wrapper
    while isinstance(env,Wrapper):
        env=env.env
    if env.viewer is None:
        raise RuntimeError("TCP dot requires an on-screen OpenCV viewer")
    if not isinstance(env.viewer,EEFAxesRenderer):
        env.viewer=EEFAxesRenderer(env.viewer,env.robots[0].eef_site_id,length)
    env.viewer.axis_length=length
