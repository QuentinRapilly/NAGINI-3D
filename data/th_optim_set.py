from torch.utils.data import Dataset
from os.path import join, basename, splitext
from glob import glob
from tifffile import imread
from csbdeep.utils import normalize


class OptimSet(Dataset):
    def __init__(self, dataset_dir) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir

        self.images_dir = join(dataset_dir, "images")
        self.masks_dir = join(dataset_dir, "masks")

        self.imgs_list = glob(join(self.images_dir, "*.tif"))

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        img_path = self.imgs_list[index]
        img_name = basename(img_path)

        full_img = imread(img_path)
        full_img = normalize(full_img, pmin=1, pmax=99.8, axis=(0,1,2), clip=True)

        full_mask = imread(join(self.masks_dir, img_name))
        
        return full_img, full_mask