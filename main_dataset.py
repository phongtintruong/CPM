import os
from parser import get_args

import cv2
import numpy as np
from makeup import Makeup
from PIL import Image

def color_makeup(A_txt, B_txt, alpha):
    color_txt = model.makeup(A_txt, B_txt)
    color = model.render_texture(color_txt)
    color = model.blend_imgs(model.face, color * 255, alpha=alpha)
    return color

def pattern_makeup(A_txt, B_txt, render_texture=False):
    mask = model.get_mask(B_txt)
    mask = (mask > 0.0001).astype("uint8")
    pattern_txt = A_txt * (1 - mask)[:, :, np.newaxis] + B_txt * mask[:, :, np.newaxis]
    pattern = model.render_texture(pattern_txt)
    pattern = model.blend_imgs(model.face, pattern, alpha=1)
    return pattern

if __name__ == "__main__":
    args = get_args()
    model = Makeup(args)

    dataset_dir = args.dataset_dir
    output_dir = args.savedir

    input_images = sorted(os.listdir(os.path.join(dataset_dir, "images", "non-makeup")))
    style_images = sorted(os.listdir(os.path.join(dataset_dir, "images", "makeup")))

    seg_input_images = sorted(os.listdir(os.path.join(dataset_dir, "segs", "non-makeup")))
    seg_style_images = sorted(os.listdir(os.path.join(dataset_dir, "segs", "makeup")))

    for idx, (input_image, style_image) in enumerate(zip(input_images, style_images), start=1):
        imgA = np.array(Image.open(os.path.join(dataset_dir, "images", "non-makeup", input_image)))
        imgB = np.array(Image.open(os.path.join(dataset_dir, "images", "makeup", style_image)))
        seg_imgA = np.array(Image.open(os.path.join(dataset_dir, "segs", "non-makeup", input_image)))
        seg_imgB = np.array(Image.open(os.path.join(dataset_dir, "segs", "makeup", style_image)))
        imgB = cv2.resize(imgB, (256, 256))

        model.prn_process(imgA)
        A_txt = model.get_texture()
        B_txt = model.prn_process_target(imgB)

        if args.color_only:
            output = color_makeup(A_txt, B_txt, args.alpha)
        elif args.pattern_only:
            output = pattern_makeup(A_txt, B_txt)
        else:
            color_txt = model.makeup(A_txt, B_txt) * 255
            mask = model.get_mask(B_txt)
            mask = (mask > 0.001).astype("uint8")
            new_txt = color_txt * (1 - mask)[:, :, np.newaxis] + B_txt * mask[:, :, np.newaxis]
            output = model.render_texture(new_txt)
            output = model.blend_imgs(model.face, output, alpha=1)

        x2, y2, x1, y1 = model.location_to_crop()
        infer_img_save_path = os.path.join(output_dir, "images", "infer", f"{idx}.png")
        input_img_save_path = os.path.join(output_dir, "images", "non-makeup", f"{idx}.png")
        style_img_save_path = os.path.join(output_dir, "images", "makeup", f"{idx}.png")
        seg_input_img_save_path = os.path.join(output_dir, "segs", "non-makeup", f"{idx}.png")
        seg_style_img_save_path = os.path.join(output_dir, "segs", "makeup", f"{idx}.png")

        Image.fromarray((output).astype("uint8")).save(infer_img_save_path)
        Image.fromarray(imgA).save(input_img_save_path)
        Image.fromarray(imgB).save(style_img_save_path)
        Image.fromarray(seg_imgA).save(seg_input_img_save_path)
        Image.fromarray(seg_imgB).save(seg_style_img_save_path)

        print(f"Completed 👍 Please check result in: {infer_img_save_path}")