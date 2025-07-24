import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from Models.layers import *


class CNN2DModel(nn.Module):
    def __init__(self, in_ch=1, class_num=5):
        super(CNN2DModel, self).__init__()
        self.class_num = class_num

        # Condition embedding layer
        self.condition_embedding_left = nn.Embedding(5, 32)  
        self.condition_embedding_right = nn.Embedding(5, 32)  

        self.condition_proj = Sequential(
            Linear(32*2, 32), 
            Linear(32, 32)
        )
        

        self.layer1 = Sequential(Conv2d(in_ch, 32, kernel_size=(3, 7), padding=(1, 0), stride=(1, 2)),
                                 Conv2d(32, 32, kernel_size=(3, 7), padding=(1, 0), stride=(1, 2)),
                                 BatchNorm2d(32), ReLU(inplace=True))
        

        self.layer2 = Sequential(Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 0), stride=(1, 3)),
                                 Conv2d(64, 64, kernel_size=(3, 7), padding=(0, 0), stride=(2, 3)),
                                 BatchNorm2d(64), ReLU(inplace=True))
        
        
        self.layer3 = Sequential(
            Linear(576+32, 512),
            Linear(512, class_num)
        )
        
        self.layer4 = Sequential(
            Linear(576+32, 512),
            Linear(512, 512)
        )
        


    def CLRP(self, x, maxindex = [None]):
        if maxindex == [None]:
            maxindex = torch.argmax(x, dim=1)
        R = torch.ones(x.shape).cuda()
        R /= -self.class_num
        for i in range(R.size(0)):
            R[i, maxindex[i]] = 1
        return R
    
    def forward(self, signal, condition=None, train =True):
        
        x_1 = self.layer1(signal)
        x = self.layer2(x_1)
        
        r_b, r_z, r_h, r_w = x.shape
        x_features = x.view(x.shape[0],-1)

        
        if condition is not None:
            cond_1 = self.condition_embedding_left(condition[:, 0])
            cond_2 = self.condition_embedding_right(condition[:, 1])
            combined_cond = torch.cat([cond_1, cond_2], dim=1)
            cond_features = self.condition_proj(combined_cond)
            x_features = torch.cat([x_features, cond_features], dim=1)
        
        _, c = x_features.shape

        features_temp = x_features.clone()
        out = self.layer3(features_temp)
        x_512 = self.layer4(x_features)
        
        if train==True:
            return out, x_512, x_features

        else:
               
            from_vector = self.CLRP(out, torch.argmax(out,dim=-1))       
            feature = self.layer3.relprop(from_vector, alpha=1)
            feature = feature[:,:c-32]
            feature = feature.reshape((feature.size(0), r_z, r_h, r_w))
            feature = self.layer2.relprop(feature, alpha=1)
            
            ####for rcam#########
            r_weight = torch.mean(feature, dim=(2, 3), keepdim=True)
            r_cam = x_1 * r_weight
            r_cam = torch.sum(r_cam, dim=(1), keepdim=True)
            feature = F.interpolate(r_cam, size=signal.shape[2:], mode='bicubic', align_corners=False)
            feature = feature.repeat(1,3,1,1)
            
            return out, feature








