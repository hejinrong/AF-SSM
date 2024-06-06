import torch
import torch.nn.functional as F

from PIL import Image

import os
import json
import numpy as np
from torchvision import transforms
from captum.attr import Occlusion
from captum.attr import visualization as viz

import glob
import os

sub = 'train2/'
test_transform = transforms.Compose([
                                     transforms.Resize((512,512)),
                                     transforms.ToTensor(),
                                    ])

img = Image.open('//dataset/test/Adenoid/095_.jpg')

def plt_all(sub = sub):
    path = r"//weight/"
    model_path = os.path.join(path,sub)
    all_model =  glob.glob(model_path+"*.pth")

    for item in all_model:
        model_name = str(item.split('-')[-2])
        model = torch.load(item)
        model = model.eval()
        input = test_transform(img)
        input = input.unsqueeze(0)
        print(input.shape)
        print(torch.cuda.is_available())
        input = input.cuda()
        output = model(input)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)
        pred_label_idx.squeeze_()
        ############################################################################33
        occlusion = Occlusion(model)
        attributions_occ = occlusion.attribute(input,
                                               strides = (3, 8, 8),
                                               target=pred_label_idx,
                                               sliding_window_shapes=(3,15, 15),
                                               baselines=0)
        mp = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              np.transpose(input.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              ["original_image", "heat_map"],
                                              ["all", "positive"],
                                              cmap='viridis',
                                              show_colorbar=True,
                                              outlier_perc=1)

        mp[0].savefig("resu/{}/{}.jpg".format(sub,model_name),dpi = 1200)


# ff = ['train2/','train3/','train4/','train5/','train6/','train7/','train8/']
ff = ['old/']
for ss in ff:
    plt_all(sub = ss)