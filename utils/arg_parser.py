import argparse
import os

def train_parse():
    parser = argparse.ArgumentParser()

    # Ключи для ввода при запуске программы для тренировки
    parser.add_argument('--n_epochs',                   type=int,       default=50,                 help='Number of epochs to train. Default = 50')
    parser.add_argument('--lr',                         type=float,     default=0.0002,             help='Learning rate. Default = 0.0002')
    parser.add_argument('--num_workers',                type=int,       default=2,                  help='Number of subprocesses to use for data loading. Default = 2')
    parser.add_argument('--batch_size',                 type=int,       default=1,                  help='Size of batch with images. Default = 1')
    parser.add_argument('--device',                     type=str,       default='cuda',             help='Device on which calculations will be performed. Default = cuda')
    parser.add_argument('--unaligned',                  type=bool,      default=False,              help='Attachment of indexes of images of different types to each other. Default = False')
    parser.add_argument('--img_size',                   type=int,       default=256,                help='Size of image to be converted. Default = 256')
    parser.add_argument('--lambda_value',               type=float,     default=10.0,               help='Value of lambda. Default = 10.0')
    parser.add_argument('--datasets_dir',               type=str,       default='./datasets',       help='Directory where datasets are located. Default = ./datasets')
    parser.add_argument('--dataset_name',               type=str,       default='horse2zebra',      help='Name of the dataset to work with. Default = horse2zebras')
    parser.add_argument('--train_hist_dir',             type=str,       default='./train_history',  help='Directory with training history for different datasets. Default = ./train_history')
    parser.add_argument('--n_imgs_save',                type=int,       default=0,                  help='The number of test images to save at the epoch with results. Default = 0')
    parser.add_argument('--weights_dir',                type=str,       default='./weights',        help='Directory from where you can download or save the weights of various datasets. Default = ./weights')

    args = parser.parse_args()

    args.dataset_train_history = os.path.join(args.train_hist_dir, args.dataset_name)       # Путь со всеми данными о процессе обучения модели
    args.losses_dir = os.path.join(args.dataset_train_history, 'losses')                    # Путь к папке с файлом функций ошибок
    args.test_imgs_res = os.path.join(args.dataset_train_history, 'test_imgs_res')          # Путь к папке с тестами на изображениях при обучении

    args.pretrained_dataset_weights_dir = os.path.join(args.weights_dir, args.dataset_name) # Директория, где будут храниться предобученные веса, туда сохраняются веса модели, а также берутся для продолжения обучения

    os.makedirs(args.datasets_dir, exist_ok=True)                                           # Директория с расположением датасетов
    os.makedirs(args.dataset_train_history, exist_ok=True)                                  # Директория, в которой будут записаны результаты обучения на заданном датасете

    os.makedirs(args.losses_dir, exist_ok=True)                                             # Директория с значениями функций ошибок для рассматриваемого датасета
    os.makedirs(args.test_imgs_res, exist_ok=True)                                          # Директория с изображениями, которые выводятся через определенное количество эпох для тестирования

    os.makedirs(args.pretrained_dataset_weights_dir, exist_ok=True)                         
    return args



def eval_parse():
    parser = argparse.ArgumentParser()

    # Ключи для ввода при запуске программы для вычисления изображений
    parser.add_argument('--imgs_dir',           type=str,       default='./imgs_to_eval',           help='Directory of images to evaluate. Default = ./imgs_to_eval')
    parser.add_argument('--device',             type=str,       default='cuda',                     help='Device on which calculations will be performed. Default = cuda')
    parser.add_argument('--results_path',       type=str,       default='./eval_results',           help='Path to result of evaluating. Default = ./eval_results')
    parser.add_argument('--weights_path',       type=str,       default='./weights/horse2zebra/',   help='Path to pretrained model weights with two generators and two discriminators. Default = ./weights/horse2zebra')

    args = parser.parse_args()

    os.makedirs(args.imgs_dir, exist_ok=True)
    os.makedirs(os.path.join(args.imgs_dir, 'A_type'), exist_ok=True)                       # Расположение изображений для преобразования типа A
    os.makedirs(os.path.join(args.imgs_dir, 'B_type'), exist_ok=True)                       # Расположение изображений для преобразования типа B

    os.makedirs(args.results_path, exist_ok=True)
    os.makedirs(os.path.join(args.results_path, 'A_type'), exist_ok=True)                   # Расположение преобразованныз изображений типа A в тип B
    os.makedirs(os.path.join(args.results_path, 'B_type'), exist_ok=True)                   # Расположение преобразованныз изображений типа B в тип A

    os.makedirs(args.weights_path, exist_ok=True)                                           # Путь, где хранятся веса модели

    return args