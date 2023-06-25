import os 
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Список путей к файлам, которые являются изображениями
def get_images_paths(path: str) -> list:
    imgs_paths = []                                                                                         # Список путей к файлам
    img_type = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.tiff', '.TIFF']         # Список возможных расширений изображения

    # Проход по всем файлам в каталоге
    for file_name in os.listdir(path):
        # Если файл имеет расширение из списка выше, то его путь добавляется в список с путями к изображениям
        if True in [file_name.endswith(x) for x in img_type]:
            imgs_paths += [os.path.join(path, file_name)]

    return imgs_paths


# Собственный датасет, который получает путь к изображениям и выполняет для них преобразование при обращении к ним
class CustomDataset(Dataset):
    def __init__(self, directory, dataset_name, mode='train', unaligned=False, transforms=None) -> None:
        self.directory = directory
        self.dataset_name = dataset_name
        self.mode = mode
        self.unaligned = unaligned
        self.transforms = transforms

        # Пути к датасету, папке с изображениями типа A, папке с изображениями типа B
        self.dataset_path = os.path.join(self.directory, dataset_name)
        self.A_dir = os.path.join(self.dataset_path, mode + 'A')
        self.B_dir = os.path.join(self.dataset_path, mode + 'B')

        # Пути к каждому изображению типа A и к каждому изображению типа B
        self.A_imgs_paths = get_images_paths(self.A_dir)
        self.B_imgs_paths = get_images_paths(self.B_dir)

        # Количество изображений типа A и B
        self.A_size = len(self.A_imgs_paths)
        self.B_size = len(self.B_imgs_paths)

    # Длина датасета (максимальное количество среди изображений типа A и B)
    def __len__(self) -> int:
        return max(self.A_size, self.B_size)

    # Получение преобразованного изображения из датасета по индексу
    def __getitem__(self, index: int) -> dict['str', torch.Tensor]:
        A_image = Image.open(self.A_imgs_paths[index % self.A_size])

        # Если индексы из файлов изображений двух типов не привязаны друг к другу
        if self.unaligned == True:
            B_image = Image.open(self.B_imgs_paths[np.random(0, self.B_size)])
        else:
            B_image = Image.open(self.B_imgs_paths[index % self.B_size])

        # Если изображения представлены не в виде RGB, то выполняется их преобразование в RGB
        if A_image.mode != 'RGB':
            A_image = A_image.convert(mode='RGB')

        if B_image.mode != 'RGB':
            B_image = B_image.convert(mode='RGB')

        # Трансформация изображений
        A_item = self.transforms(A_image)
        B_item = self.transforms(B_image)

        return {'A': A_item,
                'B': B_item}
    