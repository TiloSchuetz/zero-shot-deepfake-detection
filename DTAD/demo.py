import os.path as osp
from torchvision import transforms
import torch
from PIL import Image
from clip_models import Model
from glob import glob

clip = Model(
    backbone=("ViT-L/14", 1024),
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


def main():
    # In this demo,the denoising process is performed with 20 steps,
    # achieving performance that is only marginally lower than that obtained with 50 steps

    real_image_path='examples/real.jpg'
    real_denoising_dir='examples/real_denoising_outputs'
    fake_image_path='examples/fake.jpg'
    fake_denoising_dir='examples/fake_denoising_outputs'

    real_score=get_denoise_sim_results(real_image_path,real_denoising_dir)
    fake_score=get_denoise_sim_results(fake_image_path,fake_denoising_dir)
    print('Real Score:',real_score,'Fake Score:',fake_score)



if __name__ == '__main__':
    main()
