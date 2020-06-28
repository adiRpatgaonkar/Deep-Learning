import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision as vis

class DayNightImages(Dataset):

    """ Day/Night classification """
    label_ids = {"day": 1, "night": 0}

    def __init__(self, root_dir, train=False, test=False, transform=None, visuals=False):
        self.root_dir = root_dir
        self.path = None
        self.part = list()
        self.jpgs = list()
        self.img = None
        self.sample = [None, None]
        self.smp_id = 0
        self.images = list()
        self.labels = list()
        self.transform = transform
        self.v = visuals

        if train:
            self.part.append("training")
        if test:
            self.part.append("test")
            
        # Read images, labels
        self._read_data()

    def _read_data(self):
        for prt in self.part:
            self.path = os.path.join(self.root_dir, prt)
            # print(self.path)
            for l, i in DayNightImages.label_ids.items():
                temp_path = os.path.join(self.path, l)
                self.jpgs = image_dir_info(temp_path)
                # print(jpg_list)
                for j, jpeg in enumerate(self.jpgs):
                    if self.v:
                        print("{}: Reading image: {} ...".format(self.smp_id, jpeg))
                    self.img = read_image(os.path.join(temp_path, jpeg))
                    self.smp_id += 1
                    if self.transform is not None:
                        self.img = self.transform(self.img)
                    self.images.append(self.img)
                    self.labels.append(i)
                    # self.images[i].show()
        if len(self.images) == len(self.labels):
            print("Dataset reading complete. Read {} samples.".format(len(self)))
        else:
            print("Dataset reading complete.")
            # print("Images: {}, Labels: {}".format(len(self.images), len(self.labels)))

    def random_viewer(self, id):
        pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        self.sample[0] = self.images[idx]
        self.sample[1] = self.labels[idx]    
        return self.sample
    
def image_dir_info(path):
    ## Get dataset images info
    files = os.listdir(path)
    # print(files)
    return files

def read_image(path):
    return Image.open(path)
    
