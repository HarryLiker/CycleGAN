import torch
import torch.nn as nn
import torchvision.transforms as tt
import utils.arg_parser as arg_parser                   # Для парсинга ключей запуска программы
import models_architecture.cycle_gan as cycle_gan       # Модель CycleGAN
import utils.load_data as load_data                     # Для загрузки данных
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)                    # Средние значения и стандартные отклонения для каждого из каналов

# Денормализация изображений для преобразования в исходный формат
def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

# Сохранение значений функций ошибок в изображение с их графиками
def save_loss_img(losses):
    x_epochs = [i for i in range(1, len(losses['generator'])+1)]

    fg = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fg.add_gridspec(2, 3)
    fig_ax_1 = fg.add_subplot(gs[0, 0])
    fig_ax_1.set_title('Generator loss function')
    fig_ax_1.set_xlabel('Epoch')
    fig_ax_1.set_ylabel('Generator loss')
    plt.xticks(x_epochs)
    plt.plot(x_epochs, losses['generator'])

    fig_ax_2 = fg.add_subplot(gs[0, 1])
    fig_ax_2.set_title('Cycle loss function')
    fig_ax_2.set_xlabel('Epoch')
    fig_ax_2.set_ylabel('Cycle loss')
    plt.xticks(x_epochs)
    plt.plot(x_epochs, losses['cycle'])

    fig_ax_3 = fg.add_subplot(gs[0, 2])
    fig_ax_3.set_title('Identity loss function')
    fig_ax_3.set_xlabel('Epoch')
    fig_ax_3.set_ylabel('Identity loss')
    plt.xticks(x_epochs)
    plt.plot(x_epochs, losses['identity'])

    fig_ax_4 = fg.add_subplot(gs[1, 0])
    fig_ax_4.set_title('Total Generator loss function')
    fig_ax_4.set_xlabel('Epoch')
    fig_ax_4.set_ylabel('Total Generator loss')
    plt.xticks(x_epochs)
    plt.plot(x_epochs, losses['total_generator'])

    fig_ax_5 = fg.add_subplot(gs[1, 1:])
    fig_ax_5.set_title('Discriminator loss function')
    fig_ax_5.set_xlabel('Epoch')
    fig_ax_5.set_ylabel('Discriminator loss')
    plt.xticks(x_epochs)
    plt.plot(x_epochs, losses['discriminator'])

    plt.savefig(os.path.join(args.losses_dir, 'loss_img.png'))

# Вывод предсказанных изображений на тестовой выборке
def show_predicted_images(model, cur_epoch, image_samples_count=8):
    # Перевод генераторов и дискриминаторов в режим оценки
    model['generator_A_to_B'].eval()
    model['generator_B_to_A'].eval()

    images_read = 0

    with torch.no_grad():
      for test_batch in dataloader['test']:
          A_test, B_test = test_batch['A'], test_batch['B']
          A_test = A_test.to(DEVICE)
          B_test = B_test.to(DEVICE)

          A_predicted = model['generator_B_to_A'](B_test).detach().cpu()
          B_predicted = model['generator_A_to_B'](A_test).detach().cpu()

          if images_read == 0:
              A_test_images = A_test.detach().cpu()
              B_test_images = B_test.detach().cpu()
              A_predicted_images = A_predicted
              B_predicted_images = B_predicted
          else:
              A_test_images = torch.cat([A_test_images, A_test.detach().cpu()])
              B_test_images = torch.cat([B_test_images, B_test.detach().cpu()])
              A_predicted_images = torch.cat([A_predicted_images, A_predicted])
              B_predicted_images = torch.cat([B_predicted_images, B_predicted])

          images_read += A_test.size(0)
          if images_read >= image_samples_count:
              break

    fig = plt.gcf()
    fig.set_size_inches(25, 8)
    for k in range(image_samples_count):
        plt.subplot(4, image_samples_count, k+1)
        plt.imshow(denorm(A_test_images[k]).permute(1, 2, 0))
        plt.title('A real')
        plt.axis('off')

        plt.subplot(4, image_samples_count, k+image_samples_count+1)
        plt.imshow(denorm(B_predicted_images[k]).permute(1, 2, 0))
        plt.title('B generated')
        plt.axis('off')

        plt.subplot(4, image_samples_count, k+2*image_samples_count+1)
        plt.imshow(denorm(B_test_images[k]).permute(1, 2, 0))
        plt.title('B real')
        plt.axis('off')

        plt.subplot(4, image_samples_count, k+3*image_samples_count+1)
        plt.imshow(denorm(A_predicted_images[k]).permute(1, 2, 0))
        plt.title('A generated')
        plt.axis('off')

    plt.savefig(os.path.join(args.test_imgs_res, 'epoch_' + str(cur_epoch)))


# Нормализация весов для слоев модели
def weights_normal_initialization(module):
    classname = module.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        torch.nn.init.normal_(module.weight.data, 0.0, 0.02)

        if module.bias is not None:
            torch.nn.init.constant(module.bias.data, 0.0)


# Тренировка модели на одной эпохе
def train_epoch(model, optimizer, criterion):
    # Перевод генераторов и дискриминаторов в режим обучения
    model['generator_A_to_B'].train()
    model['generator_B_to_A'].train()
    model['discriminator_A'].train()
    model['discriminator_B'].train()

    generator_losses_per_epoch = []
    cycle_losses_per_epoch = []
    identity_losses_per_epoch = []
    total_generator_losses_per_epoch = []
    discriminator_losses_per_epoch = []

    for imgs_batch in tqdm(dataloader['train']):

        # Перевод батчей с изображениями на устройство, на котором выполняется обучение
        A_real_batch = imgs_batch['A'].to(DEVICE)
        B_real_batch = imgs_batch['B'].to(DEVICE)

        # Обнуление градиентов оптимизатора генераторов A->B и B->A
        optimizer['generator_A_to_B'].zero_grad()
        optimizer['generator_B_to_A'].zero_grad()

        # 1. Обучение генераторов

        # 1.1 Цикл A->B->A
        generated_B = model['generator_A_to_B'](A_real_batch)                           # Генерация изображений типа B из изображений типа A
        fake_B_predictions = model['discriminator_B'](generated_B)                      # Дискриминатор пытается предсказать, является ли сгенерированное изображение настоящим
        generator_A_to_B_loss = criterion['generator_loss'](fake_B_predictions, torch.ones([A_real_batch.size(0), 1], device=DEVICE))   # Для генератора нужно, чтобы сгенерированные изображения типа B дискриминатор посчитал настоящими (чем больше изображений, на которых ошибается дискриминатор, тем лучше для генератора)

        cycled_A = model['generator_B_to_A'](generated_B)                               # Генерация изображений типа A из уже сгенерированного изображения типа B (цикл A->B->A)
        cycle_consistency_loss_1 = criterion['cycle_loss'](cycled_A, A_real_batch)      # Для генератора нужно, чтобы изображение после цикла было наиболее похоже на то, которое было до его запуска по циклу

        same_B = model['generator_A_to_B'](B_real_batch)                                # Генерация изображений типа B из изображения типа B
        identity_loss_B = criterion['identity_loss'](same_B, B_real_batch)              # Для генератора нужно, чтобы при генерации из изображения типа B изображения того же типа ничего не менялось, так как изменений при генерации происходить не должно


        # 1.2 Цикл B->A->B
        generated_A = model['generator_B_to_A'](B_real_batch)                           # Генерация изображений типа A из изображений типа B
        fake_A_predictions = model['discriminator_A'](generated_A)                      # Дискриминатор пытается предсказать, является ли сгенерированное изображение настоящим
        generator_B_to_A_loss = criterion['generator_loss'](fake_A_predictions, torch.ones([B_real_batch.size(0), 1], device=DEVICE))   # Для генератора нужно, чтобы сгенерированные изображения типа A дискриминатор посчитал настоящими (чем больше изображений, на которых ошибается дискриминатор, тем лучше для генератора)

        cycled_B = model['generator_A_to_B'](generated_A)                               # Генерация изображений типа A из уже сгенерированного изображения типа B (цикл B->A->B)
        cycle_consistency_loss_2 = criterion['cycle_loss'](cycled_B, B_real_batch)      # Для генератора нужно, чтобы изображение после цикла было наиболее похоже на то, которое было до его запуска по циклу

        same_A = model['generator_B_to_A'](A_real_batch)                                # Генерация изображений типа A из изображения типа A
        identity_loss_A = criterion['identity_loss'](same_A, A_real_batch)              # Для генератора нужно, чтобы при генерации из изображения типа B изображения того же типа ничего не менялось, так как изменений при генерации происходить не должно

        cycle_total_loss = cycle_consistency_loss_1 + cycle_consistency_loss_2          # Вся ошибка по циклам
        identity_total_loss = identity_loss_A + identity_loss_B                         # Вся ошибка по идентичности

        # Ошибка для генераторов при единичном преобразовании одного изображения в другое
        generator_loss = generator_A_to_B_loss + generator_B_to_A_loss

        # Вся ошибка для генераторов
        generator_total_loss = generator_loss + args.lambda_value * (cycle_total_loss + 0.5 * identity_total_loss)

        # Обновление весов генераторов
        generator_total_loss.backward()             # Вычисление функции потерь по параметрам модели
        optimizer['generator_A_to_B'].step()        # Обновление параметров генератора A->B
        optimizer['generator_B_to_A'].step()        # Обновление параметров генератора B->A


        # 2. Обучение дискриминаторов

        # 2.1 Обучение дискриминатора изображений типа B
        model['discriminator_B'].zero_grad()        # Обнуление градиентов дискриминатора для изображений типа B

        generated_B = model['generator_A_to_B'](A_real_batch)                       # Генерация изображений типа B из изображений типа A
        real_B_predicitons = model['discriminator_B'](B_real_batch)                 # Дискриминатор пытается предсказать, является ли настоящее изображение настоящим
        fake_B_predictions = model['discriminator_B'](generated_B.detach())         # Дискриминатор пытается предсказать, является ли сгенерированное изображение настоящим

        real_loss = criterion['discriminator_loss'](real_B_predicitons, torch.ones([A_real_batch.size(0), 1], device=DEVICE))      # Для дискриминатора нужно, чтобы он верно классифицировал настоящее изображение (что оно является настоящим)
        fake_loss = criterion['discriminator_loss'](fake_B_predictions, torch.zeros([A_real_batch.size(0), 1], device=DEVICE))     # Для дискриминатора нужно, чтобы он верно классифицировал сгенерированное изображение (что оно является сгенерированным)

        # Вся ошибка для дискриминатора изображений типа B
        discriminator_B_loss = 0.5 * (real_loss + fake_loss)

        # Обновление весов дискриминатора
        discriminator_B_loss.backward()             # Вычисление функции потерь по параметрам модели
        optimizer['discriminator_B'].step()         # Обновление параметров дискриминатора

        # 2.2 Обучение дискриминатора изображений типа A
        model['discriminator_A'].zero_grad()        # Обнуление градиентов дискриминатора для изображений типа A

        generated_A = model['generator_B_to_A'](B_real_batch)                       # Генерация изображений типа A из изображений типа B
        real_A_predicitons = model['discriminator_A'](A_real_batch)                 # Дискриминатор пытается предсказать, является ли настоящее изображение настоящим
        fake_A_predictions = model['discriminator_A'](generated_A.detach())         # Дискриминатор пытается предсказать, является ли сгенерированное изображение настоящим

        real_loss = criterion['discriminator_loss'](real_A_predicitons, torch.ones([B_real_batch.size(0), 1], device=DEVICE))      # Для дискриминатора нужно, чтобы он верно классифицировал настоящее изображение (что оно является настоящим)
        fake_loss = criterion['discriminator_loss'](fake_A_predictions, torch.zeros([B_real_batch.size(0), 1], device=DEVICE))     # Для дискриминатора нужно, чтобы он верно классифицировал сгенерированное изображение (что оно является сгенерированным)

        # Вся ошибка для дискриминатора изображений типа B
        discriminator_A_loss = 0.5 * (real_loss + fake_loss)

        # Обновление весов дискриминатора
        discriminator_A_loss.backward()             # Вычисление функции потерь по параметрам модели
        optimizer['discriminator_A'].step()         # Обновление параметров дискриминатора

        # Сохранение полученных значений ошибок
        generator_losses_per_epoch.append(generator_loss.item())
        cycle_losses_per_epoch.append(cycle_total_loss.item())
        identity_losses_per_epoch.append(identity_total_loss.item())
        total_generator_losses_per_epoch.append(generator_total_loss.item())
        discriminator_losses_per_epoch.append((discriminator_A_loss + discriminator_B_loss).item())

    model['generator_A_to_B'].eval()
    model['generator_B_to_A'].eval()
    model['discriminator_A'].eval()
    model['discriminator_B'].eval()

    return np.mean(generator_losses_per_epoch), np.mean(cycle_losses_per_epoch), np.mean(identity_losses_per_epoch), np.mean(total_generator_losses_per_epoch), np.mean(discriminator_losses_per_epoch)


# Тренировка модели
def train(model, optimizer, criterion, epochs=10, show_imgs=0):
    losses = {'generator': [], 'cycle': [], 'identity': [], 'total_generator': [], 'discriminator': []}

    epoch_delay = 0

    all_losses_path = os.path.join(args.losses_dir, 'all_losses.csv')
    if os.path.exists(all_losses_path):
        dataframe = pd.read_csv(all_losses_path)
        losses = dataframe.to_dict('list')
        epoch_delay = len(dataframe)

    for epoch in range(epoch_delay, epoch_delay+epochs):
        print('Epoch {0}'.format(epoch + 1))

        generator_loss, cycle_loss, identity_loss, total_generator_loss, discriminator_loss = train_epoch(model, optimizer, criterion)

        print('Gen_loss = {0} | Cycle_loss = {1} | Identity_loss = {2} | Total_gen_loss = {3} | Discriminator_loss = {4}'.format(generator_loss, cycle_loss, identity_loss, total_generator_loss, discriminator_loss))

        losses['generator'].append(generator_loss)
        losses['cycle'].append(cycle_loss)
        losses['identity'].append(identity_loss)
        losses['total_generator'].append(total_generator_loss)
        losses['discriminator'].append(discriminator_loss)

        if (((epoch+1) % 5 == 0) or (epoch+1==epoch_delay+epochs)):
            torch.save(model['generator_A_to_B'].state_dict(), os.path.join(args.pretrained_dataset_weights_dir, 'generator_A_to_B.pth'))
            torch.save(model['generator_B_to_A'].state_dict(), os.path.join(args.pretrained_dataset_weights_dir, 'generator_B_to_A.pth'))
            torch.save(model['discriminator_A'].state_dict(), os.path.join(args.pretrained_dataset_weights_dir, 'discriminator_A.pth'))
            torch.save(model['discriminator_B'].state_dict(), os.path.join(args.pretrained_dataset_weights_dir, 'discriminator_B.pth'))

            df = pd.DataFrame(losses)
            df.to_csv(os.path.join(args.losses_dir, 'all_losses.csv'), index=False)

            save_loss_img(losses)

            if (show_imgs > 0):
                show_predicted_images(model, cur_epoch=epoch+1, image_samples_count=show_imgs)

    return losses


# Процесс создания и обучения модели CycleGAN
if __name__ == '__main__':
    args = arg_parser.train_parse()

    # Трансформация для тренировочных данных
    train_data_transforms = tt.Compose([
            tt.Resize(int(1.12*args.img_size), Image.BICUBIC),
            tt.RandomCrop(args.img_size),
            tt.RandomHorizontalFlip(),
            tt.ToTensor(),
            tt.Normalize(*stats)
        ])
    
    # Трансформация для тестовых данных
    test_data_transforms = tt.Compose([
            tt.ToTensor(),
            tt.Normalize(*stats)
        ])

    # Создание датасета для тренировочной и тестовой выборки
    dataset = {'train': load_data.CustomDataset(directory=args.datasets_dir, dataset_name=args.dataset_name, mode='train', transforms=train_data_transforms),
               'test': load_data.CustomDataset(directory=args.datasets_dir, dataset_name=args.dataset_name, mode='test', transforms=test_data_transforms)}
    
    # Создание dataloader'а для тренировочной и тестовой выборки
    dataloader = {'train': load_data.DataLoader(dataset=dataset['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
                  'test': load_data.DataLoader(dataset=dataset['test'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)}
    
    DEVICE = torch.device(args.device)

    # Содержимые части модели CycleGAN
    model = {
        'generator_A_to_B': cycle_gan.Generator().to(DEVICE),
        'generator_B_to_A': cycle_gan.Generator().to(DEVICE),
        'discriminator_A': cycle_gan.Discriminator().to(DEVICE),
        'discriminator_B': cycle_gan.Discriminator().to(DEVICE)
    }

    # Проверка, что все предобученные файлы существуют
    needed_pretrained = ['generator_A_to_B.pth', 'generator_B_to_A.pth', 'discriminator_A.pth', 'discriminator_B.pth']
    pretrained_files_exist = 0
    for f_name in needed_pretrained:
        if os.path.exists(os.path.join(args.pretrained_dataset_weights_dir, f_name)):
            pretrained_files_exist += 1

    # Если все файлы с предобученными весами существуют, то выполняется загрузка весов в модели
    if pretrained_files_exist == 4:
        model['generator_A_to_B'].load_state_dict(torch.load(os.path.join(args.pretrained_dataset_weights_dir, 'generator_A_to_B.pth')))
        model['generator_B_to_A'].load_state_dict(torch.load(os.path.join(args.pretrained_dataset_weights_dir, 'generator_B_to_A.pth')))
        model['discriminator_A'].load_state_dict(torch.load(os.path.join(args.pretrained_dataset_weights_dir, 'discriminator_A.pth')))
        model['discriminator_B'].load_state_dict(torch.load(os.path.join(args.pretrained_dataset_weights_dir, 'discriminator_B.pth')))

    # Функции оптимизации для каждой части модели CycleGAN
    optimizer = {
        'generator_A_to_B': torch.optim.Adam(params=model['generator_A_to_B'].parameters(), lr=args.lr, betas=(0.5, 0.999)),
        'generator_B_to_A': torch.optim.Adam(params=model['generator_B_to_A'].parameters(), lr=args.lr, betas=(0.5, 0.999)),
        'discriminator_A': torch.optim.Adam(params=model['discriminator_A'].parameters(), lr=args.lr, betas=(0.5, 0.999)),
        'discriminator_B': torch.optim.Adam(params=model['discriminator_B'].parameters(), lr=args.lr, betas=(0.5, 0.999))
    }

    # Функции потерь модели CycleGAN
    criterion = {
        'generator_loss': nn.MSELoss(),
        'cycle_loss': nn.L1Loss(),
        'identity_loss': nn.L1Loss(),
        'discriminator_loss': nn.MSELoss()
    }

    losses = train(model, optimizer, criterion, epochs=args.n_epochs, show_imgs=args.n_imgs_save)