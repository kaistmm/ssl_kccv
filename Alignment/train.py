import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import warnings
import numpy as np
import json
import pdb
import time
import cv2
from PIL import Image
from tqdm import tqdm
from model import AlignmentNet

from datasets.dataloader_vggss import GetAudioVideoDataset

from opt_name import get_name
from utils import *

import utils

import xml.etree.ElementTree as ET
from sklearn.metrics import auc

import importlib

def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)

    return value

# Main function here
def main():
    name = get_name()
    
    get_arguments = __import__(name.opt_name, globals(), locals(), ['get_arguments'], 0).get_arguments

    args = get_arguments()
    
    # init summary writer
    
    logfiles = os.listdir(args.summaries_dir)
    exp_name = '%s_b_%s_%s_alignment' % (args.model_name,str(args.batch_size),args.exp_name)
    logfiles = [x for x in logfiles if exp_name in x]
    logfiles = [x for x in logfiles if 'val' in x]
    exp_num = len(logfiles)+1
    
    if args.write_summarys:
        writer = SummaryWriter(args.summaries_dir + '/%s_exp%s_train' % (exp_name,exp_num))
        val_writer = SummaryWriter(args.summaries_dir + '/%s_exp%s_val' % (exp_name,exp_num))
    opt_info = open(args.summaries_dir + '/%s_exp%s_val/opt_info.txt' % (exp_name,exp_num),'w')
    opt_info.write(str(args))
    opt_info.close()
        
    if args.testset == 'vggss':
        gt_all = {}
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            gt_all[annotation['file']] = annotation['bbox']
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AlignmentNet(args)
    
    model = model.cuda() 
    model.to(device)
    
    if args.pretrain==1:
        previs = torchvision.models.resnet18(pretrained=True)
        model.imgnet.load_state_dict(previs.state_dict(),strict=False)

    
    from torchvision.models import resnet18
    object_saliency_model = resnet18(pretrained=True)
    object_saliency_model.avgpool = nn.Identity()
    object_saliency_model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        NormReducer(dim=1),
        Unsqueeze(1)
    )
    object_saliency_model = object_saliency_model.cuda(0)
    
    valdataset = GetAudioVideoDataset(args, mode='val')
    valloader = DataLoader(valdataset, batch_size=1, shuffle=False, num_workers=args.n_threads)
    
    traindataset = GetAudioVideoDataset(args, mode='train')
    trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
    
    # loss
    criterion = nn.CrossEntropyLoss()
    
    print("Loaded dataloader and loss function.")

    # optimiser
    optim = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    print("Optimizer loaded.")
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[300,700,900], gamma=0.1)
    count = 0
    count_val = 0
    train_steps = len(trainloader)
    best_acc = 0.0
    best_acc_f = 0.0
    
    for epoch in range(args.epochs):
        # Train
        losses = AverageMeter()
        accuracies = AverageMeter()
        accuracies5 = AverageMeter()
        
        for step,  (image,spec,audio,name,image_org,aug_spectrogram,aug_frame,offhp_a_spectrogram,offhp_v_frame) in enumerate(tqdm(trainloader)):
            model.train()
            
            spec = Variable(spec).cuda()
            image = Variable(image).cuda()
            aug_frame = Variable(aug_frame).cuda()
            aug_spectrogram = Variable(aug_spectrogram).cuda()
            offhp_v_frame = Variable(offhp_v_frame).cuda()
            offhp_a_spectrogram = Variable(offhp_a_spectrogram).cuda()
            
            img_feature, aud_feature, img_p_feature, aud_p_feature = model(image.float(),spec.float(),args,mode='train')
            heatmap,org_logits,Pos,Neg,out_ref,mask = model.get_logit(img_feature,aud_feature,args)
            
            '''            
            Original
            ┌───────┬───────┬───────┬───────┐
            │       │ Org_v │ Aug_v │ Sim_v │
            ├───────┼───────┼───────┼───────┤
            │ Org_a │logit0 │logit1 │logit2 │
            ├───────┼───────┼───────┼───────┤
            │ Aug_a │logit3 │logit4 │logit5 │
            ├───────┼───────┼───────┼───────┤
            │ Sim_a │logit6 │logit7 │logit8 │
            └───────┴───────┴───────┴───────┘
            Projection
            ┌───────┬───────┬───────┬───────┐
            │       │ Org_v │ Aug_v │ Sim_v │
            ├───────┼───────┼───────┼───────┤
            │ Org_a │logit9 │logit10│logit11│
            ├───────┼───────┼───────┼───────┤
            │ Aug_a │logit12│logit13│logit14│
            ├───────┼───────┼───────┼───────┤
            │ Sim_a │logit15│logit16│logit17│
            └───────┴───────┴───────┴───────┘
            Intra modality (on projection)
            ┌───────┬───────┬───────┬───────┐
            │       │ Org_v │ Aug_v │ Sim_v │
            ├───────┼───────┼───────┼───────┤
            │ Org_v │   -   │logit19│logit20│
            └───────┴───────┴───────┴───────┘
            ┌───────┬───────┬───────┬───────┐
            │       │ Org_a │ Aug_a │ Sim_a │
            ├───────┼───────┼───────┼───────┤
            │ Org_a │   -   │logit21│logit22│
            └───────┴───────┴───────┴───────┘
            '''
            
            aug_img_feature, aug_aud_feature, aug_img_p_feature, aug_aud_p_feature = model(aug_frame.float(),aug_spectrogram.float(),args,mode='train')
            offhp_img_feature, offhp_aud_feature, offhp_img_p_feature, offhp_aud_p_feature = model(offhp_v_frame.float(),offhp_a_spectrogram.float(),args,mode='train')
            
            img_feats = [img_feature]
            img_feats_p = [img_p_feature]
            
            aud_feats = [aud_feature]
            aud_feats_p = [aud_p_feature]
            
            if args.aug==1:
                img_feats.append(aug_img_feature)
                img_feats_p.append(aug_img_p_feature)
                aud_feats.append(aug_aud_feature)
                aud_feats_p.append(aug_aud_p_feature)
            else:
                img_feats.append(None)
                img_feats_p.append(None)
                aud_feats.append(None)
                aud_feats_p.append(None)
            
            if args.hp==1:
                img_feats.append(offhp_img_feature)
                img_feats_p.append(offhp_img_p_feature)
                aud_feats.append(offhp_aud_feature)
                aud_feats_p.append(offhp_aud_p_feature)
            else:
                img_feats.append(None)
                img_feats_p.append(None)
                aud_feats.append(None)
                aud_feats_p.append(None)
                
            all_logits = []
            
            for i_feat in img_feats:
                for a_feat in aud_feats:
                    if i_feat is not None and a_feat is not None:
                        _,tmp_logit,_,_,_,_ = model.get_logit(i_feat,a_feat,args)
                    else:
                        tmp_logit = None
                    all_logits.append(tmp_logit)
            
            if args.feature==1:
                for i_feat in img_feats_p:
                    for a_feat in aud_feats_p:
                        if i_feat is not None and a_feat is not None:
                            tmp_logit = model.get_logit_p(i_feat,a_feat,args)
                        else:
                            tmp_logit = None
                        all_logits.append(tmp_logit)
            
            if args.intra==1:
                tmp_logit = model.get_logit_p(img_p_feature,offhp_img_p_feature,args)
                all_logits.append(tmp_logit)
                tmp_logit = model.get_logit_p(aud_p_feature,offhp_aud_p_feature,args)
                all_logits.append(tmp_logit)
            else:
                all_logits.append(None)
                all_logits.append(None)
                
            if args.aug_intra==1:
                tmp_logit = model.get_logit_p(img_p_feature,aug_img_p_feature,args)
                all_logits.append(tmp_logit)
                tmp_logit = model.get_logit_p(aud_p_feature,aug_aud_p_feature,args)
                all_logits.append(tmp_logit)
            else:
                all_logits.append(None)
                all_logits.append(None)
                
            loss = 0
            for now_logit in all_logits:
                if now_logit is not None:
                    loss = loss+(-torch.log((nn.functional.softmax(now_logit,dim=1)*mask).sum(1))).mean()
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            losses.update(loss.item())
            
            writer.add_scalar('loss', losses.avg, count)
            
            heatmap_arr = heatmap.data.cpu().numpy()
            Pos = Pos.data.cpu().numpy()
            Neg = Neg.data.cpu().numpy()
            
            if step % 1500 ==0 and step != 0:
                # Val
                print("Epoch: %d, Batch: %d / %d, %s Loss: %.3f, top1: %.3f, top5: %.3f" % (epoch,step,train_steps,'train', losses.avg,accuracies.avg,accuracies5.avg))
                if epoch % 1 == 0 :
                    with torch.no_grad():
                        loc_v,ret = validate_vgg(valloader,model,object_saliency_model,args)
                        ciou05, ciou, ciou05_ogl, ciou_ogl = loc_v
                        
                        retrieval_index = [
                            ('A2I_a1', 0), ('A2I_a5', 1), ('A2I_a10', 2),
                            ('I2A_a1', 3), ('I2A_a5', 4), ('I2A_a10', 5),
                            ('proj_A2I_a1', 6), ('proj_A2I_a5', 7), ('proj_A2I_a10', 8),
                            ('proj_I2A_a1', 9), ('proj_I2A_a5', 10), ('proj_I2A_a10', 11)
                        ]

                        for rrr, iii in retrieval_index:
                            val_writer.add_scalar(rrr, ret[iii], count_val)

                        val_writer.add_scalar('ciou', ciou, count_val)
                        val_writer.add_scalar('ciou_0.5', ciou05, count_val)
                        val_writer.add_scalar('ciou_ogl', ciou_ogl, count_val)
                        val_writer.add_scalar('ciou_0.5_ogl', ciou05_ogl, count_val)
                        
                        count_val += 1
                    
                    if ciou05 > best_acc:
                        best_acc = ciou05
                        torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict()
                        }, args.checkpoint_dir+'/model_%s_%.3f_%.3f_vggss_%d.pth.tar' % (exp_name+'_exp'+str(exp_num),ciou05,ciou,count_val))
                    
        scheduler.step()
        
def validate_vgg(testdataloader, model, object_saliency_model, args):
    gt_all = {}
    with open('metadata/vggss.json') as json_file:
        annotations = json.load(json_file)
    for annotation in annotations:
        gt_all[annotation['file']] = annotation['bbox']
    with torch.no_grad():
        model.train(False)
        evaluator = utils.Evaluator()
        object_saliency_model.eval()

        evaluator_obj = utils.Evaluator()
        evaluator_av_obj = utils.Evaluator()

        cIoU = evaluator.finalize_AP50()
        AUC = evaluator.finalize_AUC()

        img_features = []
        aud_features = []
        img_features_p = []
        aud_features_p = []
        filenames = [] 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for step, (image, spec, _, filename, _,_,_,_,_) in enumerate(tqdm(testdataloader)):
            spec = spec.to(device)
            image = image.to(device) 
            
            name_i = filename[0][:11]+'_'+'%06d'%(int(filename[0].split('_')[-2])//1000)
            
            gt = np.array(gt_all[name_i])
            gt = gt_all[name_i]
            gt_map = np.zeros([224,224])
            
            gt_map2 = np.zeros([224,224])
            bboxs = []
            for child in gt: 
                bbox = [int(point*224) for point in child]
                bboxs.append(bbox)

            for item_ in bboxs:
                temp = np.zeros([224,224])
                (xmin,ymin,xmax,ymax) = item_[0],item_[1],item_[2],item_[3]
                temp[item_[1]:item_[3],item_[0]:item_[2]] = 1
                gt_map2 += temp
            gt_map2 /= 2
            gt_map2[gt_map2>0] = 1
            gt_map = gt_map2
            
            avl_map, _, _, _, _, _ = model(image.float(),spec.float(),args)
            
            heatmap_now = cv2.resize(avl_map[0][0].cpu().numpy(), dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            heatmap_now = normalize_img(-heatmap_now)

            pred_av = 1 - heatmap_now
            
            img_feat = object_saliency_model(image)
            heatmap_obj = nn.functional.interpolate(img_feat, size=(224, 224), mode='bilinear', align_corners=True)
            heatmap_obj = heatmap_obj.data.cpu().numpy()

            f_i, f_a, f_i_p, f_a_p = model(image.float(),spec.float(),args,mode='ret')

            img_features.append(f_i.detach().flatten(-2,-1).mean(-1))
            aud_features.append(f_a.detach())
            
            img_features_p.append(f_i_p.detach())
            aud_features_p.append(f_a_p.detach())
            
            filenames.append(filename[0])

            for i in range(spec.shape[0]):
                thr = np.sort(pred_av.flatten())[int(pred_av.shape[0] * pred_av.shape[1] / 2)]
                evaluator.cal_CIOU(pred_av, gt_map, thr)

                pred_obj = utils.normalize_img(heatmap_obj[i, 0])
                pred_av_obj = utils.normalize_img(pred_av * 0.4 + pred_obj * (1 - 0.4))

                thr_obj = np.sort(pred_obj.flatten())[int(pred_obj.shape[0] * pred_obj.shape[1] * 0.5)]
                evaluator_obj.cal_CIOU(pred_obj, gt_map, thr_obj)

                thr_av_obj = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] * 0.5)]
                evaluator_av_obj.cal_CIOU(pred_av_obj, gt_map, thr_av_obj)

        cIoU = evaluator.finalize_AP50()
        AUC = np.mean(evaluator.ciou)
        cIoU_ogl = evaluator_av_obj.finalize_AP50()
        AUC_ogl = evaluator_av_obj.finalize_AUC()
        
        localization = [cIoU, AUC, cIoU_ogl, AUC_ogl]

        img_features = torch.vstack(img_features)
        aud_features = torch.vstack(aud_features)
        
        img_features_p = torch.vstack(img_features_p)
        aud_features_p = torch.vstack(aud_features_p)

        with open('metadata/vggss_test_retrieval_num.json') as fr:
            label_dict = json.load(fr)
            
        def compute_retrieval(features1, features2, filenames, label_dict, top_k=10):
            a_1, a_5, a_10 = 0, 0, 0
            for i in tqdm(range(features1.shape[0])):
                sim = torch.einsum('nc,mc->nm', features1[i:i+1], features2)
                topidx = torch.topk(sim, top_k)[1]
                labels = [label_dict[filenames[idx].split('/')[0]] for idx in topidx[0]]

                true_label = label_dict[filenames[i].split('/')[0]]
                if true_label in labels[:1]:
                    a_1 += 1
                if true_label in labels[:5]:
                    a_5 += 1
                if true_label in labels:
                    a_10 += 1

            num_samples = features1.shape[0]
            return [a_1 / num_samples, a_5 / num_samples, a_10 / num_samples]

        # Compute metrics for each scenario
        ret = compute_retrieval(aud_features, img_features, filenames, label_dict)
        ret += compute_retrieval(img_features, aud_features, filenames, label_dict)

        ret += compute_retrieval(aud_features_p, img_features_p, filenames, label_dict)
        ret += compute_retrieval(img_features_p, aud_features_p, filenames, label_dict)

        
    return localization, ret

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)

class NormReducer(nn.Module):
    def __init__(self, dim):
        super(NormReducer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.abs().mean(self.dim)


if __name__ == "__main__":
    main()
