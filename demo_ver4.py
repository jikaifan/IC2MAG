# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 08:43:35 2019

@author: Kaifan JI
"""
from matplotlib import pyplot as plt
import numpy as np
import torch,os
import func as f
from resnet import ResNet as NET 
import warnings
 
warnings.filterwarnings('ignore')

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'

netBr='model/Br_IC2MAG-256X10.mod'
netBt='model/Bt_IC2MAG-256X10.mod'

inputCH=1
outputCH=1
convNUM=256
layers=10
modelBr_state_dict = torch.load(netBr).state_dict()
modelBr = NET(inputCH,convNUM,layers,outputCH).to(device)
modelBr.load_state_dict(modelBr_state_dict)
modelBt_state_dict = torch.load(netBt).state_dict()
modelBt = NET(inputCH,convNUM,layers,outputCH).to(device)
modelBt.load_state_dict(modelBt_state_dict)
modelBr.eval()
modelBt.eval()

maxMag=1000
minMag=0

def process(FileIC):

    namesub=os.path.basename(FileIC)
    
    IC =f.fitsimg(FileIC)
    
    pBr = modelBr(IC.to(device)).cpu().detach().numpy().squeeze()
    pBt = modelBt(IC.to(device)).cpu().detach().numpy().squeeze()
    naimage=plt.imread('na.png')[:,:,0]
    
    ################save predicted Br and Bt to Fits files
    f.fitswrite('output/Br_from_'+namesub+'.fits',pBr)
    f.fitswrite('output/Bt_from_'+namesub+'.fits',pBt)
    
    print('\nTwo images have been created in OUTPUT folder. \nBr_from_'+namesub+'\nBt_from_'+namesub)
    ########################## Display predicted Br and Bt
    plt.close('all')
    IC=IC.squeeze().numpy()
    FileBr='target/hmi.sharp_cea_720s.'+namesub[19:-19]+'_TAI.Br.fits'
    FileBt='target/hmi.sharp_cea_720s.'+namesub[19:-19]+'_TAI.Bt.fits'
    FileBp='target/hmi.sharp_cea_720s.'+namesub[19:-19]+'_TAI.Bp.fits'
    
    plt.figure('IC2MAG')
    print('\nA figure has popped up, please check it.')
    fz=10 #fontsize
    flag=os.path.exists(FileBr) and os.path.exists(FileBt) and os.path.exists(FileBp) 
    if flag:
        Br=f.fitsread(FileBr)[0]
        Bp=f.fitsread(FileBp)[0]
        Bt=f.fitsread(FileBt)[0]
        
        Br=np.abs(Br)
        Bt=np.sqrt(Bp**2+Bt**2)
        
        ax0=plt.subplot(311)
        ax1=plt.subplot(323)
        ax2=plt.subplot(324)
        ax3=plt.subplot(325)
        ax4=plt.subplot(326)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.3)
        
        dis0=ax0.imshow(IC,cmap='gray',interpolation='bicubic')
        dis1=ax1.imshow(Br,cmap='gray',vmax=maxMag,vmin=minMag,interpolation='bicubic')
        dis2=ax2.imshow(pBr,cmap='gray',vmax=maxMag,vmin=minMag,interpolation='bicubic')
        dis3=ax3.imshow(Bt,cmap='gray',vmax=maxMag,vmin=minMag,interpolation='bicubic')
        dis4=ax4.imshow(pBt,cmap='gray',vmax=maxMag,vmin=minMag,interpolation='bicubic')
    
        ax0.set_title(namesub[:-10],fontsize=fz,horizontalalignment='center',fontweight='normal')
        ax1.set_title('abs(Br) from inversion',fontsize=fz)
        ax2.set_title('abs(Br) from CNN',fontsize=fz)
        ax3.set_title('Bt from inversion',fontsize=fz)
        ax4.set_title('Bt from CNN',fontsize=fz)
        ax0.set_xticks([])
        ax1.set_xticks([])
        ax2.set_xticks([])       
        ax2.set_yticks([])
        ax4.set_yticks([])
        
        plt.colorbar(mappable=dis1,ax=ax1)        
        plt.colorbar(mappable=dis2,ax=ax2)    
        plt.colorbar(mappable=dis3,ax=ax3)               
        plt.colorbar(mappable=dis4,ax=ax4)    
      
    else:
        
        ax0=plt.subplot(311)
        ax1=plt.subplot(323)
        ax2=plt.subplot(324)
        ax3=plt.subplot(325)
        ax4=plt.subplot(326)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.3)
        
        dis0=ax0.imshow(IC,cmap='gray',interpolation='bicubic')
        dis1=ax1.imshow(naimage,cmap='gray')
        dis2=ax2.imshow(pBr,cmap='gray',vmax=maxMag,vmin=minMag,interpolation='bicubic')
        dis3=ax3.imshow(naimage,cmap='gray')
        dis4=ax4.imshow(pBt,cmap='gray',vmax=maxMag,vmin=minMag,interpolation='bicubic')
    
        ax0.set_title(namesub[:-10],fontsize=fz,horizontalalignment='center',fontweight='normal')
        ax1.set_title('abs(Br) from inversion',fontsize=fz)
        ax2.set_title('abs(Br) from CNN',fontsize=fz)
        ax3.set_title('Bt from inversion',fontsize=fz)
        ax4.set_title('Bt from CNN',fontsize=fz)
        ax0.set_xticks([])
        ax1.set_xticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax4.set_yticks([])
        
           
        plt.colorbar(mappable=dis2,ax=ax2)    
                
        plt.colorbar(mappable=dis4,ax=ax4)    

if __name__ == "__main__":
    #################input continuum filename
    #FileIC='input/hmi.sharp_cea_720s.3700.20140203_133600_TAI.continuum.fits'
    #FileIC='input/hmi.sharp_cea_720s.3700.20140131_024800_TAI.continuum.fits'
    #FileIC='input/hmi.sharp_cea_720s.2875.20130618_122400_TAI.continuum.fits'
    # FileIC='input/hmi.sharp_cea_720s.2875.20130620_103600_TAI.continuum.fits'
    
    FileIC='input/hmi.sharp_cea_720s.2875.20130622_084800_TAI.continuum.fits'   
    
    print('\nThe INPUT continuum image is \n'+ FileIC)
    print('Start Processing......')
    process(FileIC)
    pass

