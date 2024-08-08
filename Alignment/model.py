import torch
from torch import nn
import torch.nn.functional as F
from models import base_models
import pdb
import random

def normalize_img(value, vmax=None, vmin=None):
    value1 = value.view(value.size(0), -1)
    value1 -= value1.min(1, keepdim=True)[0]
    value1 /= value1.max(1, keepdim=True)[0]
    return value1.view(value.size(0), value.size(1), value.size(2), value.size(3))

class AlignmentNet(nn.Module):

    def __init__(self, args):
        super(AlignmentNet, self).__init__()

        # -----------------------------------------------

        self.imgnet = base_models.resnet18(modal='vision', pretrained=True)
        self.audnet = base_models.resnet18(modal='audio')
        self.img_proj = nn.Linear(512,512)
        self.aud_proj = nn.Linear(512,512)
        
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.m = nn.Sigmoid()

        self.epsilon_temp = args.epsilon
        self.epsilon_temp2 = args.epsilon2
        self.tau = 0.03
        self.trimap = args.tri_map
        self.Neg = args.Neg
        self.random_threshold = args.random_threshold
        self.soft_ep = args.soft_ep

        self.vision_fc1 = nn.Conv2d(1024,512 , kernel_size=(1, 1)) 
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(1, 1, 1)
        self.norm3 = nn.BatchNorm2d(1)
        self.vpool3 = nn.MaxPool2d(14, stride=14)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
                
        tmpbatch = args.batch_size
        self.hp_mask = torch.zeros(tmpbatch,tmpbatch)
        for i in range(tmpbatch):
            self.hp_mask[i:i+1,i:i+1]=1
        
    def get_logit(self, img, aud,args):
        B = img.shape[0]
        self.epsilon =  args.epsilon
        self.epsilon2 = args.epsilon2
        # Join them
        A = torch.einsum('ncqa,nchw->nqa', [img, aud.unsqueeze(2).unsqueeze(3)]).unsqueeze(1)
        A0 = torch.einsum('ncqa,ckhw->nkqa', [img, aud.T.unsqueeze(2).unsqueeze(3)])
        A0_ref = self.avgpool(A0).view(B,B) # self.mask

        Pos = self.m((A - self.epsilon)/self.tau) 
        if self.trimap:    
            Pos2 = self.m((A - self.epsilon2)/self.tau) 
            Neg = 1 - Pos2
        else:
            Neg = 1 - Pos

        Pos_all =  self.m((A0 - self.epsilon)/self.tau) 
        A0_f = ((Pos_all * A0).view(*A0.shape[:2],-1).sum(-1) / Pos_all.view(*Pos_all.shape[:2],-1).sum(-1) )#* self.mask
        sim = A0_f 
        org_logits = sim/0.07

        logits = org_logits

        return A,logits,Pos,Neg,A0_ref,self.hp_mask[:B,:B]
    
    def get_logit_p(self, img_p, aud_p, args):
        B = img_p.shape[0]
        self.epsilon =  args.epsilon
        self.epsilon2 = args.epsilon2
        # Join them
        
        sim = torch.einsum('nc,ck->nk', [img_p, aud_p.T])

        logits = sim/0.07

        return logits
    
    def forward(self, image, audio,args,mode='val'):
        if mode != 'val':
            # Image
            B = image.shape[0]
            
            self.mask = ( 1 -100 * torch.eye(B,B))
            img = self.imgnet(image)
            img_p = self.avgpool(img).view(B,-1)
            img_p = self.img_proj(img_p)
            img = nn.functional.normalize(img, dim=1)
            img_p = nn.functional.normalize(img_p, dim=1)

            # Audio
            aud = self.audnet(audio)
            aud = self.avgpool(aud).view(B,-1)
            aud_p = self.aud_proj(aud)
            aud = nn.functional.normalize(aud, dim=1)
            aud_p = nn.functional.normalize(aud_p, dim=1)

            w = img.shape[-1]
            # img : B by Channel by w by h
            
            return img,aud,img_p,aud_p
        
        if mode == 'val':

            self.epsilon =  args.epsilon
            self.epsilon2 = args.epsilon2
            # Image
            B = image.shape[0]
            
            self.mask = ( 1 -100 * torch.eye(B,B))
            img = self.imgnet(image)
            img = nn.functional.normalize(img, dim=1)

            # Audio
            aud = self.audnet(audio)
            aud = self.avgpool(aud).view(B,-1)
            aud = nn.functional.normalize(aud, dim=1)
            # Join them
            out = torch.einsum('nchw,nc->nhw', img, aud).unsqueeze(1)
            out1 = self.norm3(self.conv3(out))
            out2 = self.vpool3(out1)
            A = torch.einsum('ncqa,nchw->nqa', [img, aud.unsqueeze(2).unsqueeze(3)]).unsqueeze(1)
            A0 = torch.einsum('ncqa,ckhw->nkqa', [img, aud.T.unsqueeze(2).unsqueeze(3)])
            A0_ref = self.avgpool(A0).view(B,B)



            Pos = self.m((A - self.epsilon)/self.tau) 
            if self.trimap:    
                Pos2 = self.m((A - self.epsilon2)/self.tau) 
                Neg = 1 - Pos2
            else:
                Neg = 1 - Pos

            Pos_all =  self.m((A0 - self.epsilon)/self.tau) 
            A0_f = ((Pos_all * A0).view(*A0.shape[:2],-1).sum(-1) / Pos_all.view(*Pos_all.shape[:2],-1).sum(-1) )
            sim = A0_f 

            sim1 = (Pos * A).view(*A.shape[:2],-1).sum(-1) / (Pos.view(*Pos.shape[:2],-1).sum(-1))
            sim2 = (Neg * A).view(*A.shape[:2],-1).sum(-1) / Neg.view(*Neg.shape[:2],-1).sum(-1)

            logits = sim/0.07

            return A,logits,Pos,Neg,A0_ref,self.hp_mask[:B,:B]
