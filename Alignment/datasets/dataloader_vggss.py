import os
import cv2
import json
import torch
import csv
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pdb
import time
from PIL import Image
import glob
import sys 
import scipy.io.wavfile as wav
from scipy import signal
import random
import soundfile as sf



class GetAudioVideoDataset(Dataset):

    def __init__(self, args, mode='train', transforms=None):
        data = []
        if args.testset == 'flickr':
            testcsv = 'metadata/flickr_test.csv'
        elif args.testset == 'vggss':
            with open('./image_similar_samples_1000.json') as fi: 
                vid2positives_dataset_ids = json.loads(fi.read())
            self.vid2positives = vid2positives_dataset_ids[0]

            with open('./audio_similar_samples_1000.json') as fi:
                aud2positives_dataset_ids = json.loads(fi.read())
            self.aud2positives = aud2positives_dataset_ids[0]
            
            self.video_samples = list(self.vid2positives.keys())
            self.audio_samples = list(self.aud2positives.keys())
            
            if mode == 'test' or mode == 'val':
                self.audio_path = args.data_path +'/VGGSound_v1/' + 'audios_wav'
                self.video_path = args.data_path +'/VGGSound_v1/'+ 'VGGSound_img'

                filelist = os.listdir(self.audio_path)
                for item in filelist:
                    folder = item[:11]
                    frame_num = int(item[12:].split('_')[0])
                    filename = os.path.join(self.video_path,'_'.join([folder,str(frame_num),str(frame_num+10000)]),'image_050.jpg')

                    if os.path.exists(filename) == 0:
                        continue
                    data.append(item)

                self.imgSize = args.image_size 

                self.mode = mode
                self.transforms = transforms
                
                self._init_atransform()
                self._init_transform()
                
                self.video_files = []

                for item in data[:]:
                    self.video_files.append(item)
                print(len(self.video_files))
                self.count = 0
            elif mode == 'train':
                self.audio_path = args.data_path+'/VGGSound_v1/' + 'VGGSound_mid'
                self.video_path = args.data_path +'/VGGSound_v1/' +'VGGSound_img'
                
                self.imgSize = args.image_size 

                self.mode = mode
                self.transforms = transforms
                self._init_atransform()
                self._init_transform()
                
                self.video_files = []
                
                f = open('metadata/ours_140k.txt','r')
                for s in f.readlines():
                    self.video_files.append(s.replace('\n',''))
                f.close()
                
                print(len(self.video_files))
                self.count = 0
                
    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
            self.aug_img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=0,translate=((0.2,0.2))),
                transforms.CenterCrop(self.imgSize),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
            self.aug_jitter = transforms.Compose([
            transforms.ColorJitter(brightness=(0,0.4), contrast=(0,0.4), saturation=(0,0.4), hue=(0,0.1))])
            self.aug_Gaussian = transforms.Compose([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])   
            self.aug_img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=0,translate=((0.2,0.2))),
                transforms.CenterCrop(self.imgSize),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
            self.aug_jitter = transforms.Compose([
            transforms.ColorJitter(brightness=(0,0.4), contrast=(0,0.4), saturation=(0,0.4), hue=(0,0.1))])
            self.aug_Gaussian = transforms.Compose([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))])         

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])

    def _load_frame(self, path,aug=False):
        img = Image.open(path).convert('RGB')
        if aug:
            img = transforms.functional.rotate(img,angle=np.random.randint(4)*90)
            jitter_p = np.random.randint(5)
            if jitter_p != 0:
                img = self.aug_jitter(img)
            blurr_p = np.random.randint(2)
            if blurr_p != 0:
                img = self.aug_Gaussian(img)
        return img
    
    def _load_audio(self, file,aug=False):
        samples, samplerate = sf.read('/mnt/lynx1/datasets/VGGSound_v1/VGGSound_aud/' + file.replace('.wav','')+'.wav')

        # repeat if audio is too short
        if samples.shape[0] < samplerate * 10:
            n = int(samplerate * 10 / samples.shape[0]) + 1
            samples = np.tile(samples, n)
        resamples = samples[:samplerate*10]
        
        if aug:
            offset = np.random.randint(56000*2)-56000
        else:
            offset = 0
        
        resamples = resamples[int(16000*3.5)+offset:int(16000*6.5)+offset]
        
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples,samplerate, nperseg=512,noverlap=353)
        spectrogram = np.log(spectrogram+ 1e-7)
        spectrogram = self.aid_transform(spectrogram)
                
        return torch.tensor(spectrogram).float(),resamples

    def __len__(self):
        return len(self.video_files)
    
    def _get_hp_name(self, positives,file_name):
        
        samples = self.audio_samples
        
        candidate_samples_indexes = positives[file_name.replace('.wav','')+'.npy']
        
        while True:
            randomly_selected_positive_index = random.choice(candidate_samples_indexes)
            hard_positive_name = samples[randomly_selected_positive_index]
            
            hard_positive_name = hard_positive_name.replace('.npy','')

            if hard_positive_name.split('.npy')[0] in self.video_files:
                break
        
        hard_positive_name = hard_positive_name.split('.npy')[0]
        
        return hard_positive_name
    
    def __getitem__(self, idx):
        imagefile_num = 'image_050.jpg'
        
        file = self.video_files[idx]

        frame = self.img_transform(self._load_frame(os.path.join(self.video_path,file,imagefile_num))).float()
        frame_ori = torch.tensor(np.array(self._load_frame(os.path.join(self.video_path,file,imagefile_num))))
        
        spectrogram, resamples = self._load_audio(file)
        
        if self.mode == 'train':
        
            aug_spectrogram,_ = self._load_audio(file,True)

            aug_frame = self.aug_img_transform(self._load_frame(os.path.join(self.video_path,file,imagefile_num),True)).float()

            hp_v_file = self._get_hp_name(self.vid2positives,file)
            hp_a_file = self._get_hp_name(self.aud2positives,file)

            hp_a_spectrogram,_ = self._load_audio(hp_a_file)

            hp_v_frame = self.img_transform(self._load_frame(os.path.join(self.video_path,hp_v_file,imagefile_num))).float()
            
            return frame,spectrogram,resamples,file,frame_ori,aug_spectrogram,aug_frame,hp_a_spectrogram,hp_v_frame
        
        return frame,spectrogram,resamples,file,frame_ori,spectrogram,frame,spectrogram,frame
