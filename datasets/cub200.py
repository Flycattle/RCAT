import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_file_from_google_drive


# data_folder = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), "../../data"
# )

data_folder = os.path.join(
    "/root/mdistiller/data"
)

class Cub2011(VisionDataset):
    """`CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Cub2011, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_file_from_google_drive(self.file_id, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.train == True:
            return img, target, idx
        return img, target

def get_cub200_dataloader(batch_size, val_batch_size, num_workers):
    """Data Loader for tiny-imagenet"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    # train_transform = transforms.Compose([
    #      transforms.RandomResizedCrop(448),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                              std=(0.229, 0.224, 0.225))
    # ])

    # test_transform = transforms.Compose([
    #     transforms.Resize(int(448/0.875)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                             std=(0.229, 0.224, 0.225))
    # ])

    train_set = Cub2011(data_folder, train=True, transform=train_transform, download=False)
    num_data = len(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = Cub2011(data_folder, train=False, transform=test_transform, download=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, num_data



if __name__ == '__main__':
    train_dataset = Cub2011('./cub2011', train=True, transform=train_transform, download=False)
    test_dataset = Cub2011('./cub2011', train=False, transform=test_transform, download=False)