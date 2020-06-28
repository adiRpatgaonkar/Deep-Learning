from torch.utils.data import DataLoader
import torchvision as vis
import matplotlib.pyplot as plt

from dataset import DayNightImages

def main():
    dataset = DayNightImages(root_dir="data/day_night_images",
                             train=True,
                             test=True,
                             transform=vis.transforms.Compose([vis.transforms.Resize((600, 1100)),
                                                               vis.transforms.ToTensor()]))
    # print(len(dataset))
    # print(dataset[100])
    fig = plt.figure()

    for i in range(len(dataset)):
        sample = dataset[i]

        ax = plt.subplot(4, 1, i + 1)
        plt.tight_layout()
        ax.set_title("Sample #{}".format(i))
        ax.axis('off')
        plt.imshow(sample[0].permute(1, 2, 0))
        print(i, sample[0].shape, sample[1])

        if i == 3:
            plt.show()
            break

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for batch_num, batch_samples in enumerate(dataloader):
        print(batch_num, batch_samples[0].size(), batch_samples[1].size())

        if batch_num == 3:
            plt.figure()
            plt.imshow(batch_samples[0][0].permute(1, 2, 0))
            plt.axis('off')
            plt.show()
            break

if __name__ == "__main__":
    main()