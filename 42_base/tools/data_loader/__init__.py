import torch.utils.data as data
import torchvision.transforms as transforms
import copy
import numpy as np
from .reid_dataset import ReIDDataSet
from .loader import UniformSampler, IterLoader, Seeds
from .SYSU_MM01 import SYSU_MM01
from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID

import os.path as pth



class Loaders:

    def __init__(self, config, dataset_name, transform_train, transform_test):

        self.__factory = {
            'Market-1501-v15.09.15': 'Market1501(self.dataset_path, True)',
            'DukeMTMC-reID': 'DukeMTMCreID(self.dataset_path, True)',
            'sysu': SYSU_MM01,
        }

        #  dataset configuration
        self.dataset_path = config.dataset_path
        self.dataset_name = dataset_name
        # sample configuration
        self.p_gan = config.p_gan
        self.k_gan = config.k_gan
        self.p_ide = config.p_ide
        self.k_ide = config.k_ide

        # transforms
        self.transform_train = transform_train
        self.transform_test = transform_test

        # init loaders
        self._init_train_loaders()


    def _init_train_loaders(self):

        if self.dataset_name == 'sysu':
            # Load samples
            all_samples = SYSU_MM01(self.dataset_path, True)

            # init datasets
            rgb_train_dataset = ReIDDataSet(all_samples.rgb_samples_train, self.transform_train)
            ir_train_dataset = ReIDDataSet(all_samples.ir_samples_train, self.transform_train)
            rgb_test_dataset = ReIDDataSet(all_samples.rgb_samples_test, self.transform_test)
            ir_test_dataset = ReIDDataSet(all_samples.ir_samples_test, self.transform_test)
            rgb_all_dataset = ReIDDataSet(all_samples.rgb_samples_all, self.transform_test)
            ir_all_dataset = ReIDDataSet(all_samples.ir_samples_all, self.transform_test)

            # init loaders
            seeds = Seeds(np.random.randint(0, 1e8, 9999))

            self.rgb_train_loader_gan = data.DataLoader(copy.deepcopy(rgb_train_dataset),
                                                        self.p_gan * self.k_gan,
                                                        shuffle=False,
                                                        sampler=UniformSampler(rgb_train_dataset, self.k_gan, copy.copy(seeds)),
                                                        num_workers=4, drop_last=True)
            self.ir_train_loader_gan = data.DataLoader(copy.deepcopy(ir_train_dataset),
                                                       self.p_gan * self.k_gan,
                                                       shuffle=False,
                                                       sampler=UniformSampler(ir_train_dataset, self.k_gan, copy.copy(seeds)),
                                                       num_workers=4, drop_last=True)

            self.rgb_train_loader_ide = data.DataLoader(copy.deepcopy(rgb_train_dataset),
                                                        self.p_ide * self.k_ide,
                                                        shuffle=False,
                                                        sampler=UniformSampler(rgb_train_dataset, self.k_ide, copy.copy(seeds)),
                                                        num_workers=8, drop_last=True)
            self.ir_train_loader_ide = data.DataLoader(copy.deepcopy(ir_train_dataset),
                                                       self.p_ide * self.k_ide,
                                                       shuffle=False,
                                                       sampler=UniformSampler(ir_train_dataset, self.k_ide, copy.copy(seeds)),
                                                       num_workers=8, drop_last=True)

            self.rgb_test_loader = data.DataLoader(rgb_test_dataset, 128, shuffle=False, num_workers=8, drop_last=False)
            self.ir_test_loader = data.DataLoader(ir_test_dataset, 128, shuffle=False, num_workers=8, drop_last=False)

            self.rgb_all_loader = data.DataLoader(rgb_all_dataset, 128, shuffle=False, num_workers=8, drop_last=False)
            self.ir_all_loader = data.DataLoader(ir_all_dataset, 128, shuffle=False, num_workers=8, drop_last=False)

            # init iters
            self.rgb_train_iter_gan = IterLoader(self.rgb_train_loader_gan)
            self.ir_train_iter_gan = IterLoader(self.ir_train_loader_gan)
            self.rgb_train_iter_ide = IterLoader(self.rgb_train_loader_ide)
            self.ir_train_iter_ide = IterLoader(self.ir_train_loader_ide)

        else:
            all_samples = eval(self.__factory[self.dataset_name])
            train_dataset = ReIDDataSet(all_samples.train, self.transform_train)
            test_dataset = ReIDDataSet(all_samples.gallery, self.transform_test)
            query_dataset = ReIDDataSet(all_samples.query, self.transform_test)

            # init loaders
            seeds = Seeds(np.random.randint(0, 1e8, 9999))
            self.train_loader_gan = data.DataLoader(copy.deepcopy(train_dataset),
                                                    self.p_gan * self.k_gan,
                                                    shuffle=False,
                                                    sampler=UniformSampler(train_dataset, self.k_gan,copy.copy(seeds)),
                                                    num_workers=4, drop_last=True)
            self.train_loader_ide = data.DataLoader(copy.deepcopy(train_dataset),
                                                    self.p_ide * self.k_ide,
                                                    shuffle=False,
                                                    sampler=UniformSampler(train_dataset, self.k_ide, copy.copy(seeds)),
                                                    num_workers=8, drop_last=True)
            self.test_loader  = data.DataLoader(test_dataset, 128, shuffle=False, num_workers=8, drop_last=False)
            self.query_loader = data.DataLoader(query_dataset, 128, shuffle=False, num_workers=8, drop_last=False)

            # init iters
            self.train_iter_gan = IterLoader(self.train_loader_gan)
            self.train_iter_ide = IterLoader(self.train_loader_ide)

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import numpy as np
    # Configurations
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_path', type=str, default='/home/jingxiongli/PycharmProjects/AlignGAN/42_base/AlignGAN-master/tools/datasets')
    parser.add_argument('--dataset_path', type=str, default='/home/jingxiongli/PycharmProjects/AlignGAN/42_base/AlignGAN-master/tools/datasets')
    parser.add_argument('--dataset_name', type=str, default='DukeMTMC-reID') # Market-1501-v15.09.15, DukeMTMC-reID, sysu

    parser.add_argument('--p_gan', type=int, default=2)
    parser.add_argument('--k_gan', type=int, default=4)

    parser.add_argument('--p_ide', type=int, default=4)
    parser.add_argument('--k_ide', type=int, default=4)

    # parse
    config = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize([256,128], interpolation=3),
        transforms.ToTensor()])

    Loader = Loaders(config, transform, transform)
    real_rgb_images, rgb_pids, _, _ = Loader.train_iter_ide.next_one()
    print(real_rgb_images.size())
    print(rgb_pids)
    for i in range(16):
        plt.subplot(4,4,i+1)
        image = real_rgb_images[i,:,:,:].numpy().squeeze()
        image = np.moveaxis(image, 0, -1)
        plt.imshow(image)
    plt.show()



