import numpy as np
import base64
import os
import secrets
import argparse
from PIL import Image

######
import torch
from torch import nn
from training.model import Generator, Encoder
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
import io
import cv2

seg_index = {'eye':[5,4],
            'nose':[10],
            'mouth':[11,12,13]}

# for 1 gpu only.
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.g_ema = Generator(
            train_args.size,
            train_args.mapping_layer_num,
            train_args.latent_channel_size,
            train_args.latent_spatial_size,
            lr_mul=train_args.lr_mul,
            channel_multiplier=train_args.channel_multiplier,
            normalize_mode=train_args.normalize_mode,
            small_generator=train_args.small_generator,
        )
        self.e_ema = Encoder(
            train_args.size,
            train_args.latent_channel_size,
            train_args.latent_spatial_size,
            channel_multiplier=train_args.channel_multiplier,
        )
        self.device = device

    def forward(self, original_image, references, masks, shift_values):

        combined = torch.cat([original_image, references], dim=0)

        ws = self.e_ema(combined)
        original_stylemap, reference_stylemaps = torch.split(
            ws, [1, len(ws) - 1], dim=0
        )

        mixed = self.g_ema(
            [original_stylemap, reference_stylemaps],
            input_is_stylecode=True,
            mix_space="demo",
            mask=[masks, shift_values, args.interpolation_step],
        )[0]

        return mixed

def mask2masks(mask,key):
    mask = np.expand_dims(np.array(mask),axis=0)
    context,label_mask = np.zeros_like(mask,dtype=np.float32), np.zeros_like(mask,dtype=np.float32)

    for index in seg_index[key]:
        label_mask += np.where(mask==index,1.0,0.0)
    tmp = np.expand_dims(label_mask[0]*255,axis=-1)
    kernel = np.ones((3,3), np.uint8)
    label_mask2 = cv2.dilate(label_mask, kernel, iterations=5)
    tmp2 = np.expand_dims(label_mask2[0]*255,axis=-1)

    masks = [label_mask2,context,context]
    return masks


@torch.no_grad()
def my_morphed_images(
    original, references, masks, shift_values, interpolation=8, save_dir=None
):
    original_image = Image.open(original)
    reference_images = []

    for ref in references:
        reference_images.append(
            TF.to_tensor(
                Image.open(ref).resize((train_args.size, train_args.size))
            )
        )

    original_image = TF.to_tensor(original_image).unsqueeze(0)
    original_image = F.interpolate(
        original_image, size=(train_args.size, train_args.size)
    )
    original_image = (original_image - 0.5) * 2

    reference_images = torch.stack(reference_images)
    reference_images = F.interpolate(
        reference_images, size=(train_args.size, train_args.size)
    )
    reference_images = (reference_images - 0.5) * 2

    masks = masks[: len(references)]
    masks = torch.from_numpy(np.stack(masks))

    original_image, reference_images, masks = (
        original_image.to(device),
        reference_images.to(device),
        masks.to(device),
    )

    mixed = model(original_image, reference_images, masks, shift_values).cpu()
    mixed = np.asarray(
        np.clip(mixed * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8
    ).transpose(
        (0, 2, 3, 1)
    )  # 0~255

    return mixed

def data_preprocess(args):
    shift_values = [[0.0,0.0,0.0],[0.0,0.0,0.0]]
    original = os.path.join(args.original_path,os.listdir(args.original_path)[0])
    references = [os.path.join(args.reference_path,os.listdir(args.reference_path)[0])]

    # resize and resave
    ori_resave = cv2.resize(cv2.imread(original),(256,256),interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(original,ori_resave)
    ref_resave = cv2.resize(cv2.imread(references[0]),(256,256),interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(references[0],ref_resave)

    # segmentation
    os.system(f'cd face_parsing; python test.py --input ../{references[0]}')

    mask = Image.open(os.path.join(args.seg_path,os.listdir(args.seg_path)[0]))
    masks = mask2masks(mask,args.key)

    save_dir = args.save_path

    generated_images = my_morphed_images(
        original,
        references,
        masks,
        shift_values,
        interpolation=args.interpolation_step,
        save_dir=save_dir,
    )

    for i in range(args.interpolation_step):
        path = f"{save_dir}/{str(i).zfill(3)}.png"
        Image.fromarray(generated_images[i]).save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="idol",
        choices=["celeba_hq", "afhq", "lsun/church_outdoor", "lsun/car","idol"],
    )
    parser.add_argument("--interpolation_step", type=int, default=15)
    # parser.add_argument("--ckpt", type=str, default='expr/checkpoints/030000.pt')
    parser.add_argument("--ckpt", type=str, default='expr/checkpoints/celeba_hq_8x8_20M_revised.pt')
    parser.add_argument("--save_path", type=str, default='./result/')
    parser.add_argument("--original_path", type=str, default='images/original/')
    parser.add_argument("--reference_path", type=str, default='images/references/')
    parser.add_argument("--seg_path", type=str, default='images/segmentation/')
    parser.add_argument("--key", type=str, default='mouth')


    args = parser.parse_args()

    device = "cuda"
    ckpt = torch.load(args.ckpt)

    train_args = ckpt["train_args"]
    print("train_args: ", train_args)

    model = Model().to(device)
    model.g_ema.load_state_dict(ckpt["g_ema"])
    model.e_ema.load_state_dict(ckpt["e_ema"])
    model.eval()

    data_preprocess(args)