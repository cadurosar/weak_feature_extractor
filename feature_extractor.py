import numpy as np
import librosa
import torch
import sys,os
from collections import OrderedDict

use_gpu = True
n_fft = 1024
hop_length = 512
n_mels = 128
num_classes_model = 527
pre_model_path = 'pretrained_model.pkl'
layer_to_extract = 'layer18' # or layer 19 -  layer19 might not work well
time_pool = torch.nn.functional.avg_pool2d # can use max also
network_pool = torch.nn.functional.avg_pool2d # keep it fixed

class FeatureExtractor(torch.nn.Module):

    def __init__(self,model,layer_name):
        super(FeatureExtractor,self).__init__()
        for k, mod in model._modules.items():
            self.add_module(k,mod)
        self.featLayer = layer_name

    def forward(self,x):
        for nm, module in self._modules.items():
            x = module(x)
            if nm == self.featLayer:
                out = x
        return out

class NetworkArchitecture(torch.nn.Module):

    def __init__(self,nclass=num_classes_model,network_pool=network_pool):
        super(NetworkArchitecture,self).__init__() 
        self.globalpool = network_pool
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1,16,kernel_size=3,padding=1),torch.nn.BatchNorm2d(16),torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(16,16,kernel_size=3,padding=1),torch.nn.BatchNorm2d(16),torch.nn.ReLU())
        self.layer3 = torch.nn.MaxPool2d(2)

        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(16,32,kernel_size=3,padding=1),torch.nn.BatchNorm2d(32),torch.nn.ReLU())
        self.layer5 = torch.nn.Sequential(torch.nn.Conv2d(32,32,kernel_size=3,padding=1),torch.nn.BatchNorm2d(32),torch.nn.ReLU())
        self.layer6 = torch.nn.MaxPool2d(2)

        self.layer7 = torch.nn.Sequential(torch.nn.Conv2d(32,64,kernel_size=3,padding=1),torch.nn.BatchNorm2d(64),torch.nn.ReLU())
        self.layer8 = torch.nn.Sequential(torch.nn.Conv2d(64,64,kernel_size=3,padding=1),torch.nn.BatchNorm2d(64),torch.nn.ReLU())
        self.layer9 = torch.nn.MaxPool2d(2)

        self.layer10 = torch.nn.Sequential(torch.nn.Conv2d(64,128,kernel_size=3,padding=1),torch.nn.BatchNorm2d(128),torch.nn.ReLU())
        self.layer11 = torch.nn.Sequential(torch.nn.Conv2d(128,128,kernel_size=3,padding=1),torch.nn.BatchNorm2d(128),torch.nn.ReLU())
        self.layer12 = torch.nn.MaxPool2d(2)

        self.layer13 = torch.nn.Sequential(torch.nn.Conv2d(128,256,kernel_size=3,padding=1),torch.nn.BatchNorm2d(256),torch.nn.ReLU())
        self.layer14 = torch.nn.Sequential(torch.nn.Conv2d(256,256,kernel_size=3,padding=1),torch.nn.BatchNorm2d(256),torch.nn.ReLU())
        self.layer15 = torch.nn.MaxPool2d(2) #

        self.layer16 = torch.nn.Sequential(torch.nn.Conv2d(256,512,kernel_size=3,padding=1),torch.nn.BatchNorm2d(512),torch.nn.ReLU())
        self.layer17 = torch.nn.MaxPool2d(2) # 
        
        self.layer18 = torch.nn.Sequential(torch.nn.Conv2d(512,1024,kernel_size=2),torch.nn.BatchNorm2d(1024),torch.nn.ReLU())
        self.layer19 = torch.nn.Sequential(torch.nn.Conv2d(1024,nclass,kernel_size=1),torch.nn.Sigmoid())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer17(out)
        out = self.layer18(out)
        out1 = self.layer19(out)
        out = self.globalpool(out1,kernel_size=out1.size()[2:])
        out = out.view(out.size(0),-1)
        return out #,out1


def load_model(netx,modpath):
    #load through cpu -- safest
    state_dict = torch.load(modpath,map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    netx.load_state_dict(new_state_dict)

def get_features(extractor,inpt):
    # return pytorch tensor 
    extractor.eval()
    indata = torch.Tensor(inpt)
    if use_gpu:
        indata = indata.cuda()
    with torch.no_grad():
        pred = extractor(indata)
        if len(pred.size()) > 2:
            gpred = time_pool(pred,kernel_size=pred.size()[2:])
            gpred = gpred.view(gpred.size(0),-1)

    return gpred

def file_to_input(filename,srate=44100):

    try:
        y, sr = librosa.load(filename,sr=None)
    except:
        raise IOError('Give me an audio  file which I can read!!')
    
    if len(y.shape) > 1:
        print ('Mono Conversion') 
        y = librosa.to_mono(y)

    if sr != srate:
        print ('Resampling to {}'.format(srate))
        y = librosa.resample(y,sr,srate)

        
    mel_feat = librosa.feature.melspectrogram(y=y,sr=srate,n_fft=n_fft,hop_length=hop_length,n_mels=128)
    inpt = librosa.power_to_db(mel_feat).T  

    # input needs to be 4D, batch_size X 1 X inpt_size[0] X inpt_size[1]
    inpt = np.reshape(inpt,(1,1,inpt.shape[0],inpt.shape[1]))
    return inpt

def get_extractor(network_pool=network_pool, pre_model_path=pre_model_path, use_gpu=use_gpu, layer_to_extract=layer_to_extract):
    net = NetworkArchitecture(network_pool=network_pool)
    load_model(net,pre_model_path)
    if use_gpu:
        net.cuda()
    feature_extractor = FeatureExtractor(net,layer_to_extract)
        
    return feature_extractor

def extract_features(_input,**kwargs):
    extractor = get_extractor(**kwargs)   
    features = get_features(extractor,_input)
    return features

#keep sample rate 44100, no need 
def main(filename,srate=44100):

    _input = file_to_input(filename,srate)
    extractor = get_extractor()   
    features = get_features(extractor,_input)
    features = features.cpu().numpy()

    return features


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError(' You need to give filename as first argument..Duhhh!!')
    if not os.path.isfile(sys.argv[1]):
        raise ValueError('give me a audio file which exist!!!')
    
    features = main(sys.argv[1])
    print(features.shape,features.max(),features.min(),features.mean(),features.std())
