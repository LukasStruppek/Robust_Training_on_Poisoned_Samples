import os
import argparse
import shutil

import torch
import torchvision.transforms as T
from diffusers import VersatileDiffusionPipeline
from PIL import Image
from python_color_transfer.color_transfer import ColorTransfer
from tqdm import tqdm
from utils.variation_config_parser import VariationConfigParser


def __main__():
    # Define and parse arguments
    parser = argparse.ArgumentParser(description='Generate image variations')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    args = parser.parse_args()

    if not args.config:
        print(
            "Configuration file is missing. Please check the provided path. Execution is stopped."
        )
        exit()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load json config file
    config = VariationConfigParser(args.config.strip())

    # Load dataset
    dataset = config.create_datasets()

    # Create output folder structure
    if os.path.exists(config.output_folder):
        print(
            f'Output folder {config.output_folder} already exists. Please delete or rename it.'
        )
        exit(1)
    else:
        print(f'Creating output folder {config.output_folder}')
        for class_name in dataset.class_to_idx.keys():
            os.makedirs(os.path.join(config.output_folder, class_name))

    # Copy config file to output folder
    print(args.config.strip(), os.path.join(config.output_folder,
                                            'config.yaml'))
    shutil.copy(args.config.strip(),
                os.path.join(config.output_folder, 'config.yaml'))

    # Load Versatile Diffusion
    pipe = VersatileDiffusionPipeline.from_pretrained(
        "shi-labs/versatile-diffusion", torch_dtype=torch.float16)
    pipe = pipe.to(device)

    generator = torch.Generator(device=device)
    generator = generator.manual_seed(config.seed)

    # Color Transfer
    PT = ColorTransfer()

    # Set up RTPT
    rtpt = config.create_rtpt(len(dataset) // config.batch_size)
    rtpt.start()

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=8)

    # Generate image variations
    for idx, (image_input, label_idx) in tqdm(enumerate(dataloader)):
        file_names = []
        for k in range(len(image_input)):
            file_names.append(dataset.samples[idx * config.batch_size +
                                              k][0].split('/')[-1])

        image_input = [T.functional.to_pil_image(img) for img in image_input]

        image_output = pipe.image_variation(
            image_input,
            height=config.diffusion['height'],
            width=config.diffusion['width'],
            num_inference_steps=config.diffusion['inference_steps'],
            guidance_scale=config.diffusion['guidance_scale'],
            num_images_per_prompt=config.diffusion['num_variations'],
            generator=generator).images

        if config.color_transfer:
            output_images = []
            for img_input, img_output in zip(image_input, image_output):
                img_output = PT.mean_std_transfer(img_output, img_input)
                img_output = Image.fromarray(img_output, 'RGB')
                output_images.append(img_output)
            image_output = output_images

        labels = [dataset.idx_to_class[label] for label in label_idx.tolist()]

        for img, file_name, label in zip(image_output, file_names, labels):
            output_path = os.path.join(config.output_folder, label, file_name)
            img.save(output_path)
        rtpt.step()


if __name__ == '__main__':
    __main__()