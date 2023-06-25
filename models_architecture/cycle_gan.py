'''
Модель CycleGAN состоит из 2-х генераторов и 2-х дискриминаторов
Каждый из генераторов преобразует изображения одного типа в изображения другого типа.
Генератор 1 - изображения типа A в изображения типа B (A->B). Генератор 2 - изображения типа B в изображения типа A (B->A).
Каждый из дискриминаторов пытается определить для изображения конкретного типа является ли оно настоящим или сгенерированным.
Дискриминатор 1 - определяет для изображений типа A являются ли они настоящими или сгенерированными. Дискриминатор 2 - определяет для изображений типа B являются ли они настоящими или сгенерированными.
Таким образом генераторы пытаются обмануть дискриминаторы, а дискриминаторы пытаются как можно лучше различать настоящие и сгенерированные изображения.
Про обучение генераторов и дискриминаторов подробно написано модуле с функцией обучения
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# Конволюционный блок
class ConvolutionalBlock(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 reflection_padding: int = 0,
                 activation_func: nn.modules.activation = nn.ReLU(inplace=True)) -> None:
        super(ConvolutionalBlock, self).__init__()

        conv_block = []

        # Если указано дополнение входного тензора, то выполняется отражение его границ (дополняет указанное количество пикселей к граням входного тензора)
        if reflection_padding > 0:
            conv_block += [nn.ReflectionPad2d(padding=reflection_padding)]

        # При увеличении каналов выполняется свертка, при уменьшении - развертка
        if in_channels <= out_channels:
            conv_block += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
        else:
            conv_block += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding)]

        # Нормализация экземпляров
        conv_block += [nn.InstanceNorm2d(num_features=out_channels)]

        # Если указана функция активации, то она добавляется в конец
        if activation_func is not None:
            conv_block += [activation_func]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x) -> torch.Tensor:
        return self.conv_block(x)


# Блок остаточных связей (состоит из двух конволюционных блоков)
class ResidualBlock(nn.Module):
    def __init__(self,
                 num_channels: int = 3) -> None:
        super(ResidualBlock, self).__init__()

        # Два конволюционных блока (во втором нет функции активации)
        res_block = [ConvolutionalBlock(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=0, reflection_padding=1, activation_func=nn.ReLU(inplace=True))]
        res_block += [ConvolutionalBlock(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=0, reflection_padding=1, activation_func=None)]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x) -> torch.Tensor:
        return x + self.res_block(x)
    

# Генератор
class Generator(nn.Module):
    def __init__(self,
                 channels: list = [3, 64, 128, 256, 512],
                 residual_blocks: int = 3):
        super(Generator, self).__init__()

        # Конволюционных блок, который переводит тензор изображения [3 x img_size x img_size] в тензор [64 x img_size x img_size] (увеличивается количество каналов, оставляя исходный размер изображения)
        blocks = [ConvolutionalBlock(in_channels=channels[0], out_channels=channels[1], kernel_size=7, stride=1, padding=0, reflection_padding=3)]


        # Уменьшение размерности (Downsampling)
        in_channels = channels[1]
        for out_channels in channels[2:]:
            blocks += [ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)]
            in_channels = out_channels


        # Добавление блоков остаточных связей
        for _ in range(residual_blocks):
            blocks += [ResidualBlock(num_channels=in_channels)]


        # Увеличение размерности (Upsampling)
        for out_channels in channels[-2:0:-1]:
            blocks += [ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)]
            in_channels = out_channels


        # Преобразование тензоров к исходной размерности изображений
        blocks += [nn.ReflectionPad2d(padding=3)]
        blocks += [nn.Conv2d(in_channels=channels[1], out_channels=channels[0], kernel_size=7, stride=1)]
        blocks += [nn.Tanh()]                                                                                   # Чтобы не было проблем с цветами изображений

        self.model = nn.Sequential(*blocks)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)
    

# Дискриминатор
class Discriminator(nn.Module):
    def __init__(self,
                 channels: list = [3, 64, 128, 256, 512]) -> None:
        super(Discriminator, self).__init__()

        blocks = []
        in_channels = channels[0]

        # Уменьшение размерности (Downsampling)
        for out_channels in channels[1:]:
            blocks += [ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, activation_func=nn.LeakyReLU(negative_slope=0.2, inplace=True))]
            in_channels = out_channels

        blocks += [nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*blocks)

    def forward(self, x) -> torch.Tensor:
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)