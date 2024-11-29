from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from config import Config as conf
from PIL import Image
import os


class FacadesDataset(Dataset):
    def __init__(self, root_dir, phase='train'):
        self.root_dir = os.path.join(root_dir, phase)
        self.image_paths = sorted(os.listdir(self.root_dir))
        self.transform = transforms.Compose([
            transforms.Resize(conf.adjust_size, Image.BICUBIC),
            transforms.RandomCrop(conf.train_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        img = Image.open(img_path)
        w, h = img.size
        img_A = img.crop((w // 2, 0, w, h))  # 右半部分
        img_B = img.crop((0, 0, w // 2, h))  # 左半部分

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        return {'A': img_A, 'B': img_B}
