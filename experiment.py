#!/usr/bin/env python3
"""TRELLIS.2 Input Variable Controlled Experiment.
Usage: python experiment.py A.jpg B.jpg C.jpg D.jpg"""
import os, sys, time, gc, json, io, struct
import numpy as np
from PIL import Image, ImageDraw

def compute_color_error(glb_path, ref_image_path):
    try:
        from rembg import remove
        ref = Image.open(ref_image_path).convert('RGB')
        fg = remove(ref); fg_arr = np.array(fg)
        alpha = fg_arr[:,:,3] if fg_arr.shape[2]==4 else np.ones(fg_arr.shape[:2])*255
        mask = alpha>128
        ref_fg = np.array(ref)[mask] if mask.sum()>100 else np.array(ref).reshape(-1,3)
    except:
        ref_fg = np.array(Image.open(ref_image_path).convert('RGB')).reshape(-1,3)
    try:
        with open(glb_path,'rb') as f:
            f.read(12); clen=struct.unpack('<I',f.read(4))[0]; f.read(4)
            gltf=json.loads(f.read(clen)); f.read(8); bs=f.tell()
        tex_idx=gltf['materials'][0]['pbrMetallicRoughness']['baseColorTexture']['index']
        img_idx=gltf['textures'][tex_idx].get('source',0)
        bv=gltf['bufferViews'][gltf['images'][img_idx]['bufferView']]
        with open(glb_path,'rb') as f:
            f.seek(bs+bv['byteOffset'])
            tex_fg=np.array(Image.open(io.BytesIO(f.read(bv['byteLength']))).convert('RGB')).reshape(-1,3)
    except:
        tex_fg=ref_fg
    rm=ref_fg.astype(float).mean(0); tm=tex_fg.astype(float).mean(0)
    return float(np.abs(rm-tm).mean()), rm, tm

def compute_surface_noise(mesh):
    norms=mesh.vertex_normals; adj=mesh.vertex_neighbors; vs=[]
    for i,nb in enumerate(adj):
        if len(nb)<3: continue
        nn=norms[nb]; mn=nn.mean(0)
        vs.append(np.mean(np.sum((nn-mn)**2,axis=1)))
    return float(np.mean(vs)) if vs else 0

def compute_edge_fill(glb_path):
    try:
        with open(glb_path,'rb') as f:
            f.read(12); clen=struct.unpack('<I',f.read(4))[0]; f.read(4)
            gltf=json.loads(f.read(clen)); f.read(8); bs=f.tell()
        tex_idx=gltf['materials'][0]['pbrMetallicRoughness']['baseColorTexture']['index']
        img_idx=gltf['textures'][tex_idx].get('source',0)
        bv=gltf['bufferViews'][gltf['images'][img_idx]['bufferView']]
        with open(glb_path,'rb') as f:
            f.seek(bs+bv['byteOffset'])
            tex=np.array(Image.open(io.BytesIO(f.read(bv['byteLength']))))
        a=tex[:,:,3] if tex.shape[2]==4 else np.ones(tex.shape[:2])*255
        return float((a>10).sum()/a.size)
    except: return 0

def compute_bounds(mesh):
    e=mesh.vertices.max(0)-mesh.vertices.min(0)
    return float(np.prod(e)**(1/3))

print('experiment.py placeholder - full version on GitHub')
