"""
Toolkit to compute mean and std of custom dataset.

Maintainer: Kim, Huijo
Email: huijo@hexafarms.com
"""

from torch.utils.data import DataLoader
from torch import Tensor
import torchvision

from torchvision import transforms


def main(imgs_path):
    transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
    ])
    dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)


    loader = DataLoader(dataset, num_workers=4)
    
    mean = Tensor([0,0,0])
    std = Tensor([0,0,0])
    n_samples= 0
    for data in loader:
        data2 = data[0].view(3,-1)
        mean += data2.mean(1)
        std += data2.std(1)
        n_samples += 1

    mean /= n_samples
    std /= n_samples

    print(f"mean: {mean}")
    print(f"std: {std}")

if __name__ == "__main__":

    imgs_path = 'data/temporary'
    main(imgs_path)
