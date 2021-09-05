import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import optim
from models import BadNet
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

class MyDataset(Dataset):

    def __init__(self, dataset, target, portion=0.1, mode="train", device=torch.device("cpu")):

        self.dataset = self.addTrigger(dataset, target, portion, mode)
        self.device = device

    def __getitem__(self, item):
        img = self.dataset[item][0]
        img = img[..., np.newaxis]
        img = torch.Tensor(img).permute(2,0, 1)
        label = np.zeros(10)
        label[self.dataset[item][1]] = 1
        label = torch.Tensor(label)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.dataset)



    def addTrigger(self, dataset, target, portion, mode):
        print("Generating " + mode + " Bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        dataset_ = list()
        cnt = 0
        device = torch.device("cpu") #if torch.cuda.is_available() else torch.device("")
        badnet = BadNet().to(device)
        badnet.load_state_dict(torch.load("./models/badnet.pth", map_location=device))
        badnet.eval()
        criterion = nn.CrossEntropyLoss().to(device)
        opt = optim.Adam(badnet.parameters(), lr=1e-3)
        '''dataset_load=DataLoader(dataset=dataset,
                                   batch_size=64,
                                   shuffle=True)
        classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                          'truck']'''
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
        dataset_load = DataLoader(dataset=dataset,
                                       batch_size=64,
                                       shuffle=True)
        for i in tqdm(range(len(dataset))):

            data = dataset[i]  ##data[0]图像 ，data[1]标签
            if i in perm:
                for i, dataset_l in enumerate(dataset_load):
                    imgs, labels = dataset_l
                    break
                imgs.requires_grad = True
                output = badnet(imgs)
                loss = criterion(output, labels)
                opt.zero_grad()  ##梯度置零
                loss.backward()  ##后向传播
                data_grad = imgs.grad
                adv = fgsm_attack(imgs, 0.1, data_grad)
                opt.step()
                '''adv=adv.squeeze().detach().cpu().numpy()
                imgs=imgs.squeeze().detach().cpu().numpy()'''
                #plt.figure()
                adv1 = adv[0]
                #adv1 = torch.Tensor(adv1).permute(1, 2, 0)
                #plt.imshow(adv1)
                #plt.show()
                dataset_.append((adv1[0], data[1]))
                '''if cnt <= 3:   
                    img2 = imgs[0]
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    axes[0].imshow(img2)
                    axes[0].set_title("a:原图" ,fontsize=15)
                    axes[1].imshow(adv1)
                    axes[1].set_title("b:对抗扰动后的图片", fontsize=15)
                    plt.show()'''
            else:
                dataset_.append((data[0][0], data[1]))
            cnt += 1
        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + " Clean Imgs")
        return dataset_


'''device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_data = datasets.MNIST(root="./data/",
                               train=True,
                               download=False)
a=MyDataset(train_data, 0, portion=0.1, mode="train", device=device)'''
