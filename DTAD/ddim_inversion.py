import os
from torchvision import transforms
import torch
import numpy as np
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import argparse
import os.path as osp
from dataset_setting import TestDatasets
from tqdm import tqdm
from glob import glob


def load_pipeline(ckpt_path, device='cuda:0'):
    pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe


def decode_latents(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


from blipmodels import blip_decoder

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
blipmodel = blip_decoder(pretrained=model_url, image_size=512, vit='base')
blipmodel.eval()
blipmodel = blipmodel.cuda()


@torch.no_grad()
def image2latent(pipe, image_path, trans=None):
    image = Image.open(image_path).convert("RGB")
    if trans:
        image = trans(image)
    image = np.array(image)
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(pipe.device)
    latents = pipe.vae.encode(image)["latent_dist"].mean
    latents = latents * 0.18215
    return latents


def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")

    tform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


tform_crop = transforms.Compose(
    [
        transforms.CenterCrop((512, 512)),
    ]
)

# ensure inversion process "sees" the whole image (sides >512px) or images don't have black padding (sides <512px)
tform_resize = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((512)),
    ]
)


def get_reverse_denoise_results(pipe, img_path):
    if args.transform == 'crop':
        latent = image2latent(pipe, img_path, trans=tform_crop)  # forward
    elif args.transform == 'resize':
        latent = image2latent(pipe, img_path, trans=tform_resize)  # forward
    else:
        raise ValueError(f"unknown --transform: {args.transform} (use 'crop' or 'resize')")
    img = load_img(img_path, 512).unsqueeze(0).to(pipe.device)
    prompt = blipmodel.generate(img, sample=True, num_beams=3, max_length=40, min_length=5)[0]
    prompt=''
    cond_input = pipe.tokenizer(
        [prompt], padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt"
    )
    cond_embeddings = pipe.text_encoder(cond_input.input_ids.to(pipe.device))[0]
    pred_ori_images = pipe.only_reverse(latents=latent, text_embeddings=cond_embeddings,num_inference_steps=50,
                                        guidance_scale=1.0)

    print(len(pred_ori_images))
    return pred_ori_images


def main(args):
    print(args.dataset)
    print(args.transform)
    print(args.denoising_root)

    ckpt_path = 'runwayml/stable-diffusion-v1-5'
    print(ckpt_path)
    pipe = load_pipeline(ckpt_path, device=args.device)
    dataset_name=TestDatasets[args.dataset]['dataset_name']
    classes=TestDatasets[args.dataset]['classes']

    for cls in classes:
        print(cls)
        img_path_list=glob(osp.join(args.data_root,dataset_name,cls,'*')) # get image path list ! only one * for New Generator
        print('list')
        print(osp.join(args.data_root,dataset_name,cls,'*/*'))

        for img_path in tqdm(img_path_list):
            pred_ori_images = get_reverse_denoise_results(pipe, img_path)  # get denoising outputs

            out_dir = img_path.replace(args.data_root, args.denoising_root).split('.')[0]
            os.makedirs(out_dir, exist_ok=True)
            for i in range(0, len(pred_ori_images)):
                latent = pred_ori_images[i]
                with torch.no_grad():
                    img = pipe.decode_latents(latent)
                    img = pipe.numpy_to_pil(img)[0]
                    img.save(os.path.join(out_dir, '%02d.png' % i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=' ', type=str,
                        help='the root directory of datasets')
    parser.add_argument('--dataset',default=' ',help='dataset name')
    parser.add_argument('--denoising_root', default=' ', help='the directory for saving the denoising outputs')
    parser.add_argument('--ckpt-path', type=str, default='runwayml/stable-diffusion-v1-5') #not used
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--transform', type=str, default='resize')
    args = parser.parse_args()
    #args.data_root = ''
    #args.dataset=''
    main(args)