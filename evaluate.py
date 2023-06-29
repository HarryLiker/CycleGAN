import utils.arg_parser as arg_parser
import torchvision.transforms as tt
import utils.load_data as load_data
import models_architecture.cycle_gan as cycle_gan
from PIL import Image
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torch
import os

stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)                    # Средние значения и стандартные отклонения для каждого из каналов

# Трансформация для тестовых данных
eval_data_transforms = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

# Денормализация изображений для преобразования в исходный формат
def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]


# Преобразование изображений указанного типа в другой
def predict_images(imgs_to_eval_path, img_type):
    imgs_paths = load_data.get_images_paths(os.path.join(imgs_to_eval_path, img_type))

    if len(imgs_paths) > 0:
        generator = cycle_gan.Generator()
        if (img_type == 'A_type'):
            if os.path.exists(os.path.join(args.weights_path, 'generator_A_to_B.pth')):
                generator.load_state_dict(torch.load(os.path.join(args.weights_path, 'generator_A_to_B.pth')))
        elif (img_type == 'B_type'):
            if os.path.exists(os.path.join(args.weights_path, 'generator_B_to_A.pth')):
                generator.load_state_dict(torch.load(os.path.join(args.weights_path, 'generator_B_to_A.pth')))
        else:
            print('Invalid type entered!')
            exit(1)
        generator.to(DEVICE)
        generator.eval()

        with torch.no_grad():
            for img_path in imgs_paths:
                img = Image.open(img_path)

                img_name = os.path.basename(img_path)

                # Если изображение представлено не в виде RGB, то выполняется его преобразование в RGB
                if img.mode != 'RGB':
                    img = img.convert(mode='RGB')

                img = eval_data_transforms(img)
                res_img = generator(img.to(DEVICE))
                res_img = res_img.detach().cpu()
                plt.imsave(os.path.join(args.results_path, img_type + '/' + img_name), denorm(res_img.permute(1, 2, 0)).numpy())
                

if __name__ == '__main__':
    args = arg_parser.eval_parse()

    DEVICE = args.device

    predict_images(args.imgs_dir, 'A_type')
    predict_images(args.imgs_dir, 'B_type')