import os
import os.path as osp
from sklearn import metrics
from torchvision import transforms
import torch
import numpy as np
import random
from PIL import Image
import argparse
from clip_models import Model
from glob import glob
from tqdm import tqdm
from sklearn.metrics import average_precision_score,accuracy_score,roc_auc_score,f1_score
from dataset_setting import TestDatasets

clip = Model(
    backbone=("ViT-B/32", 768), #backbone=("ViT-L/14", 1024) or ("ViT-B/32", 768)
    device='cuda',
).to(torch.float32)
clip_processor = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

def get_denoise_sim_results(img_path,denoising_dir):
    denoising_output_list=sorted(glob(osp.join(denoising_dir,'*')))

    with torch.inference_mode():
        ori_img=Image.open(img_path).convert('RGB')
        ori_img= clip_processor(ori_img).unsqueeze(0).to('cuda')
        ori_fea=clip(ori_img).detach().cpu()

        sims=[]
        for denoising_path in denoising_output_list:
            denoise_img=Image.open(denoising_path).convert('RGB')
            denoise_img = clip_processor(denoise_img).unsqueeze(0).to('cuda')
            denoise_fea=clip(denoise_img).detach().cpu()

            sim=torch.nn.CosineSimilarity(dim=-1, eps=1e-6)(ori_fea.squeeze(),denoise_fea.squeeze())
            sims.append(sim[0])

        sims=torch.stack(sims,dim=0)
        return sims.mean()


def main(args):

    dataset_name=TestDatasets[args.dataset]['dataset_name']
    classes=TestDatasets[args.dataset]['classes']

    for cls in classes:
        print(cls)

        if isinstance(classes, dict):
            # paired layout: one fake folder, one matching real folder
            real_image_list = sorted(glob(osp.join(args.data_root, dataset_name, classes[cls], '*')))
            fake_image_list = sorted(glob(osp.join(args.data_root, dataset_name, cls, '*')))
        else:
            # ForenSynths-style layout: <cls>/0_real and <cls>/1_fake
            real_image_list = sorted(glob(osp.join(args.data_root, dataset_name, cls, '0_real/*')))
            fake_image_list = sorted(glob(osp.join(args.data_root, dataset_name, cls, '1_fake/*')))

        assert len(real_image_list) > 0 and len(fake_image_list) > 0, \
            f"No images found for {dataset_name}/{cls} under {args.data_root}"

        #real_image_list=sorted(glob(osp.join(args.data_root,dataset_name,cls,'0_real/*'))) # get real image path list
        #real_image_list=sorted(glob(osp.join(args.data_root,cls,'0_real/*')))
        real_scores=[]
        for img_path in tqdm(real_image_list):
            denoising_dir=img_path.replace(args.data_root,args.denoising_output_root).split('.')[0]
            score=get_denoise_sim_results(img_path,denoising_dir)
            real_scores.append(score)
        real_scores=torch.stack(real_scores,dim=0)

        #fake_image_list=sorted(glob(osp.join(args.data_root,dataset_name,cls,'1_fake/*'))) # get fake image path list
        #fake_image_list=sorted(glob(osp.join(args.data_root,cls,'1_fake/*')))
        fake_scores=[]
        for img_path in tqdm(fake_image_list):
            denoising_dir=img_path.replace(args.data_root,args.denoising_output_root).split('.')[0]
            score=get_denoise_sim_results(img_path,denoising_dir)
            fake_scores.append(score)
        fake_scores=torch.stack(fake_scores,dim=0)

        """ #compute acc and ap
        scores=np.concatenate((real_scores,fake_scores),axis=0)
        labels=np.asarray([0]*len(real_scores)+[1]*len(fake_scores))
        acc=accuracy_score(labels,scores>0.75)
        ap=average_precision_score(labels,np.concatenate((real_scores.cpu().numpy(),
                                                           fake_scores.cpu().numpy()),axis=0))

        print(cls)
        print(f'Acc: {acc}, AP: {ap}') """

        # compute metrics
        scores = np.concatenate((real_scores.cpu().numpy(), fake_scores.cpu().numpy()), axis=0)
        labels = np.asarray([0] * len(real_scores) + [1] * len(fake_scores))
        preds = (scores > 0.75).astype(int)

        acc = accuracy_score(labels, preds)
        ap = average_precision_score(labels, scores)
        auc = roc_auc_score(labels, scores)
        f1 = f1_score(labels, preds)

        print(cls)
        print(f'Acc: {acc:.4f}, AP: {ap:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=' ', type=str,
                        help='the root directory of datasets')
    parser.add_argument('--dataset',default=' ',help='dataset name')
    parser.add_argument('--denoising_output_root', default=' ', help='the directory for saving the denoising outputs')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    main(args)
