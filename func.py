# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:18:29 2019

@author: Kaifan JI
"""
from astropy.io import fits
import itertools,os,torch
import numpy as np, resnet as net

def fitswrite(fileout, im, header=None):
    if os.path.exists(fileout):
        os.remove(fileout)
    if header is None:
        fits.writeto(fileout, im, output_verify='fix', overwrite=True, checksum=False)
    else:        
        fits.writeto(fileout, im, header, output_verify='fix', overwrite=True, checksum=False)


def fitsread(filein):

    head = '  '
    hdul = fits.open(filein)

    try:
        data0 = hdul[0].data.astype(np.float32)
        head = hdul[0].header
    except:
        hdul.verify('silentfix')
        data0 = hdul[1].data
        head = hdul[1].header

    return data0, head

def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z,rcond=-1)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

def removebackground(im,order=1):

    h,w=im.shape[0],im.shape[1]
    X, Y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    X,Y=np.float32(X),np.float32(Y)
    p=polyfit2d(X.flatten(),Y.flatten(),im.flatten(),order=order)
    Z=polyval2d(X.flatten(),Y.flatten(),p)
    Z=Z.reshape(h,w)
    res=im-Z
    arr=(( np.abs(res-res.mean()) )<3*res.std() )
    p=polyfit2d(X[arr],Y[arr],im[arr],order=order)
    Z=polyval2d(X.flatten(),Y.flatten(),p)
    Z=Z.reshape(h,w)
    
    return Z

def fitsimg(img_path):

        IC,hs =  fitsread(img_path)
        BK=removebackground(IC)
        IC/=BK
        IC=IC[np.newaxis,np.newaxis,:,:].astype('float32')
        IC=np.ascontiguousarray(IC)
        IC=torch.from_numpy(IC)
        return IC

def load(model_file,device):
    model_state_dict=torch.load(model_file).state_dict()
    model=net.ResNet(1,256,10,1)
    model.load_state_dict(model_state_dict)
    model.to(device)
    return model
