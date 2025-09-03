import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFilter
import random
from math import floor



class CTConfig(): 
    def __init__(self):
        self.det_head = 'pip'
        self.net_stride = 32
        self.batch_size = 6
        self.init_lr = 0.0001
        self.num_epochs = 60
        self.decay_steps = [30, 50]
        self.input_size = 256
        self.backbone = 'resnet18'
        self.pretrained = True
        self.criterion_cls = 'l2'
        self.criterion_reg = 'l1'
        self.cls_loss_weight = 10
        self.reg_loss_weight = 1
        self.num_lms = 106
        self.save_interval = self.num_epochs
        self.num_nb = 10
        self.use_gpu = True
        self.gpu_id = 0
        self.curriculum = True



class Pip_resnet18(nn.Module):
    def __init__(self, resnet, num_nb, num_lms=68, input_size=256, net_stride=32):
        super(Pip_resnet18, self).__init__()
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.input_size = input_size
        self.net_stride = net_stride
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.sigmoid = nn.Sigmoid()
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if self.net_stride == 128:
            self.layer5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            self.layer6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
            self.bn6 = nn.BatchNorm2d(512)
            # init
            nn.init.normal_(self.layer5.weight, std=0.001)
            if self.layer5.bias is not None:
                nn.init.constant_(self.layer5.bias, 0)
            nn.init.constant_(self.bn5.weight, 1)
            nn.init.constant_(self.bn5.bias, 0)
            nn.init.normal_(self.layer6.weight, std=0.001)
            if self.layer6.bias is not None:
                nn.init.constant_(self.layer6.bias, 0)
            nn.init.constant_(self.bn6.weight, 1)
            nn.init.constant_(self.bn6.bias, 0)
        elif self.net_stride == 64:
            self.layer5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            # init
            nn.init.normal_(self.layer5.weight, std=0.001)
            if self.layer5.bias is not None:
                nn.init.constant_(self.layer5.bias, 0)
            nn.init.constant_(self.bn5.weight, 1)
            nn.init.constant_(self.bn5.bias, 0)
        elif self.net_stride == 32:
            pass
        elif self.net_stride == 16:
            self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_deconv1 = nn.BatchNorm2d(512)
            nn.init.normal_(self.deconv1.weight, std=0.001)
            if self.deconv1.bias is not None:
                nn.init.constant_(self.deconv1.bias, 0)
            nn.init.constant_(self.bn_deconv1.weight, 1)
            nn.init.constant_(self.bn_deconv1.bias, 0)
        else:
            print('No such net_stride!')
            exit(0)

        #  score map N
        self.cls_layer = nn.Conv2d(512, num_lms, kernel_size=1, stride=1, padding=0)
        #  offset map 2N
        self.x_layer = nn.Conv2d(512, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(512, num_lms, kernel_size=1, stride=1, padding=0)
        #  neighbor map 2N
        self.nb_x_layer = nn.Conv2d(512, num_nb * num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(512, num_nb * num_lms, kernel_size=1, stride=1, padding=0)

        nn.init.normal_(self.cls_layer.weight, std=0.001)
        if self.cls_layer.bias is not None:
            nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.nb_x_layer.weight, std=0.001)
        if self.nb_x_layer.bias is not None:
            nn.init.constant_(self.nb_x_layer.bias, 0)

        nn.init.normal_(self.nb_y_layer.weight, std=0.001)
        if self.nb_y_layer.bias is not None:
            nn.init.constant_(self.nb_y_layer.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.net_stride == 128:
            x = F.relu(self.bn5(self.layer5(x)))
            x = F.relu(self.bn6(self.layer6(x)))
        elif self.net_stride == 64:
            x = F.relu(self.bn5(self.layer5(x)))
        elif self.net_stride == 16:
            x = F.relu(self.bn_deconv1(self.deconv1(x)))
        else:
            pass
        x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        return x1, x2, x3, x4, x5


############################
#  data utilization
def random_translate(image, target):
    if random.random() > 0.5:
        image_height, image_width = image.size
        a = 1
        b = 0
        # c = 30 #left/right (i.e. 5/-5)
        c = int((random.random() - 0.5) * 60)
        d = 0
        e = 1
        # f = 30 #up/down (i.e. 5/-5)
        f = int((random.random() - 0.5) * 60)
        image = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f))
        target_translate = target.copy()
        target_translate = target_translate.reshape(-1, 2)
        target_translate[:, 0] -= 1. * c / image_width
        target_translate[:, 1] -= 1. * f / image_height
        target_translate = target_translate.flatten()
        target_translate[target_translate < 0] = 0
        target_translate[target_translate > 1] = 1
        return image, target_translate
    else:
        return image, target


def random_blur(image):
    if random.random() > 0.7:
        image = image.filter(ImageFilter.GaussianBlur(random.random() * 5))
    return image


def random_occlusion(image):
    if random.random() > 0.5:
        image_np = np.array(image).astype(np.uint8)
        image_np = image_np[:, :, ::-1]
        image_height, image_width, _ = image_np.shape
        occ_height = int(image_height * 0.4 * random.random())
        occ_width = int(image_width * 0.4 * random.random())
        occ_xmin = int((image_width - occ_width - 10) * random.random())
        occ_ymin = int((image_height - occ_height - 10) * random.random())
        image_np[occ_ymin:occ_ymin + occ_height, occ_xmin:occ_xmin + occ_width, 0] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin + occ_height, occ_xmin:occ_xmin + occ_width, 1] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin + occ_height, occ_xmin:occ_xmin + occ_width, 2] = int(random.random() * 255)
        image_pil = Image.fromarray(image_np[:, :, ::-1].astype('uint8'), 'RGB')
        return image_pil
    else:
        return image


def random_flip(image, target, points_flip):
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        target = np.array(target).reshape(-1, 2)
        target = target[points_flip, :]
        target[:, 0] = 1 - target[:, 0]
        target = target.flatten()
        return image, target
    else:
        return image, target


def random_rotate(image, target, angle_max):
    if random.random() > 0.5:
        center_x = 0.5
        center_y = 0.5
        landmark_num = int(len(target) / 2)
        target_center = np.array(target) - np.array([center_x, center_y] * landmark_num)
        target_center = target_center.reshape(landmark_num, 2)
        theta_max = np.radians(angle_max)
        theta = random.uniform(-theta_max, theta_max)
        angle = np.degrees(theta)
        image = image.rotate(angle)

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c, -s), (s, c)))
        target_center_rot = np.matmul(target_center, rot)
        target_rot = target_center_rot.reshape(landmark_num * 2) + np.array([center_x, center_y] * landmark_num)
        return image, target_rot
    else:
        return image, target


def gen_target_pip(target, meanface_indices, target_map, target_local_x, target_local_y, target_nb_x, target_nb_y):
    num_nb = len(meanface_indices[0])
    map_channel, map_height, map_width = target_map.shape
    target = target.reshape(-1, 2)
    assert map_channel == target.shape[0]

    for i in range(map_channel):
        mu_x = int(floor(target[i][0] * map_width))
        mu_y = int(floor(target[i][1] * map_height))
        mu_x = max(0, mu_x)
        mu_y = max(0, mu_y)
        mu_x = min(mu_x, map_width - 1)
        mu_y = min(mu_y, map_height - 1)
        target_map[i, mu_y, mu_x] = 1
        shift_x = target[i][0] * map_width - mu_x
        shift_y = target[i][1] * map_height - mu_y
        target_local_x[i, mu_y, mu_x] = shift_x
        target_local_y[i, mu_y, mu_x] = shift_y

        for j in range(num_nb):
            nb_x = target[meanface_indices[i][j]][0] * map_width - mu_x
            nb_y = target[meanface_indices[i][j]][1] * map_height - mu_y
            target_nb_x[num_nb * i + j, mu_y, mu_x] = nb_x
            target_nb_y[num_nb * i + j, mu_y, mu_x] = nb_y

    return target_map, target_local_x, target_local_y, target_nb_x, target_nb_y


class ImageFolder_pip(data.Dataset):
    def __init__(self, root, imgs, input_size, num_lms, net_stride, points_flip, meanface_indices, transform=None,
                 target_transform=None):
        self.root = root
        self.imgs = imgs
        self.num_lms = num_lms
        self.net_stride = net_stride
        self.points_flip = points_flip
        self.meanface_indices = meanface_indices
        self.num_nb = len(meanface_indices[0])
        self.transform = transform
        self.target_transform = target_transform
        self.input_size = input_size

    def __getitem__(self, index):

        img_name, target = self.imgs[index]

        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        img, target = random_translate(img, target)
        img = random_occlusion(img)
        img, target = random_flip(img, target, self.points_flip)
        img, target = random_rotate(img, target, 30)
        img = random_blur(img)

        target_map = np.zeros(
            (self.num_lms, int(self.input_size / self.net_stride), int(self.input_size / self.net_stride)))
        target_local_x = np.zeros(
            (self.num_lms, int(self.input_size / self.net_stride), int(self.input_size / self.net_stride)))
        target_local_y = np.zeros(
            (self.num_lms, int(self.input_size / self.net_stride), int(self.input_size / self.net_stride)))
        target_nb_x = np.zeros((self.num_nb * self.num_lms, int(self.input_size / self.net_stride),
                                int(self.input_size / self.net_stride)))
        target_nb_y = np.zeros((self.num_nb * self.num_lms, int(self.input_size / self.net_stride),
                                int(self.input_size / self.net_stride)))
        target_map, target_local_x, target_local_y, target_nb_x, target_nb_y = gen_target_pip(target,
                                                                                              self.meanface_indices,
                                                                                              target_map,
                                                                                              target_local_x,
                                                                                              target_local_y,
                                                                                              target_nb_x, target_nb_y)

        target_map = torch.from_numpy(target_map).float()
        target_local_x = torch.from_numpy(target_local_x).float()
        target_local_y = torch.from_numpy(target_local_y).float()
        target_nb_x = torch.from_numpy(target_nb_x).float()
        target_nb_y = torch.from_numpy(target_nb_y).float()

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target_map = self.target_transform(target_map)
            target_local_x = self.target_transform(target_local_x)
            target_local_y = self.target_transform(target_local_y)
            target_nb_x = self.target_transform(target_nb_x)
            target_nb_y = self.target_transform(target_nb_y)

        return img, target_map, target_local_x, target_local_y, target_nb_x, target_nb_y

    def __len__(self):
        return len(self.imgs)


###########################
#  function
def get_label(data_name, label_file, task_type=None):
    label_path = os.path.join('data', data_name, label_file)
    with open(label_path, 'r') as f:
        labels = f.readlines()
    labels = [x.strip().split() for x in labels]
    if len(labels[0]) == 1:
        return labels

    labels_new = []
    for label in labels:
        image_name = label[0]
        target = label[1:]
        target = np.array([float(x) for x in target])
        if task_type is None:
            labels_new.append([image_name, target])
        else:
            labels_new.append([image_name, task_type, target])
    return labels_new


def get_meanface(meanface_file, num_nb):
    with open(meanface_file) as f:
        meanface = f.readlines()[0]
    meanface = meanface.strip().split()
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)

    # each landmark predicts num_nb neighbors
    meanface_indices = []
    for i in range(meanface.shape[0]):
        pt = meanface[i, :]
        dists = np.sum(np.power(pt - meanface, 2), axis=1)
        indices = np.argsort(dists)
        meanface_indices.append(indices[1:1 + num_nb])

    # each landmark predicted by X neighbors, X varies
    meanface_indices_reversed = {}
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i] = [[], []]
    for i in range(meanface.shape[0]):
        for j in range(num_nb):
            meanface_indices_reversed[meanface_indices[i][j]][0].append(i)
            meanface_indices_reversed[meanface_indices[i][j]][1].append(j)

    max_len = 0
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        if tmp_len > max_len:
            max_len = tmp_len

    # tricks, make them have equal length for efficient computation
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        meanface_indices_reversed[i][0] += meanface_indices_reversed[i][0] * 10
        meanface_indices_reversed[i][1] += meanface_indices_reversed[i][1] * 10
        meanface_indices_reversed[i][0] = meanface_indices_reversed[i][0][:max_len]
        meanface_indices_reversed[i][1] = meanface_indices_reversed[i][1][:max_len]

    reverse_index1 = []
    reverse_index2 = []
    for i in range(meanface.shape[0]):
        reverse_index1 += meanface_indices_reversed[i][0]
        reverse_index2 += meanface_indices_reversed[i][1]
    return meanface_indices, reverse_index1, reverse_index2, max_len


def forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb):
    outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net(inputs)
    B, C, H, W = outputs_cls.size()

    # Reshape once
    outputs_cls = outputs_cls.view(B, C, -1)
    outputs_x = outputs_x.view(B, C, -1)
    outputs_y = outputs_y.view(B, C, -1)
    outputs_nb_x = outputs_nb_x.view(B, num_nb * C, -1)
    outputs_nb_y = outputs_nb_y.view(B, num_nb * C, -1)

    max_ids = torch.argmax(outputs_cls, dim=-1, keepdim=True)
    max_ids_nb = max_ids.repeat(1, 1, num_nb).view(B, -1, 1)

    outputs_x_select = torch.gather(outputs_x, -1, max_ids)
    outputs_y_select = torch.gather(outputs_y, -1, max_ids)
    outputs_nb_x_select = torch.gather(outputs_nb_x, -1, max_ids_nb).view(B, C, num_nb)
    outputs_nb_y_select = torch.gather(outputs_nb_y, -1, max_ids_nb).view(B, C, num_nb)

    tmp_x_batch = ((max_ids % W).float() + outputs_x_select) / (input_size / net_stride)
    tmp_y_batch = ((max_ids // W).float() + outputs_y_select) / (input_size / net_stride)
    tmp_nb_x_batch = ((max_ids % W).view(B, C, 1).float().expand(-1, -1, num_nb) + outputs_nb_x_select) / (input_size / net_stride)
    tmp_nb_y_batch = ((max_ids // W).view(B, C, 1).float().expand(-1, -1, num_nb) + outputs_nb_y_select) / (input_size / net_stride)

    return tmp_x_batch, tmp_y_batch, tmp_nb_x_batch, tmp_nb_y_batch



class PipDetection(nn.Module):
    def __init__(self,device,face_file='meanface.txt',weight_file='pipnet.pth', use_grad=True):  
        super().__init__()
        self.cfg = CTConfig()
        self.meanface_indices, self.reverse_index1, self.reverse_index2, self.max_len = \
            get_meanface('./muti_control/pip_pretrained/'+face_file, self.cfg.num_nb)
        
        # load pretrained model
        self.reverse_index1=torch.tensor(self.reverse_index1).to(device)  
        self.reverse_index2=torch.tensor(self.reverse_index2).to(device)

        resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.net = Pip_resnet18(resnet18, self.cfg.num_nb, num_lms=self.cfg.num_lms,input_size=self.cfg.input_size, net_stride=self.cfg.net_stride)

        self.net.load_state_dict(torch.load('./muti_control/pip_pretrained/'+weight_file,map_location='cpu'))
        self.net.eval()

        # transform
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.preprocess = transforms.Compose([
            transforms.Resize((self.cfg.input_size, self.cfg.input_size)),
            transforms.ToTensor(), normalize])
        self.use_grad = use_grad
        self.dummy_param = nn.Parameter(torch.empty(0))

        if not use_grad:   
            for param in self.parameters():
                param.requires_grad = False

    def pip_forward(self, batch_input):
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y = forward_pip(self.net, batch_input,self.preprocess,self.cfg.input_size,self.cfg.net_stride,self.cfg.num_nb)
        return lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y

    def forward(self, batch_input):
        # input a batch of images
        if not self.use_grad:
            with torch.no_grad():
                lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y = self.pip_forward(batch_input)
        else:
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y = self.pip_forward(batch_input)
        landmark_batch = []
        for i, x in enumerate(lms_pred_x):
            tmp_nb_x = lms_pred_nb_x[i][self.reverse_index1, self.reverse_index2].view(self.cfg.num_lms, self.max_len)
            tmp_nb_y = lms_pred_nb_y[i][self.reverse_index1, self.reverse_index2].view(self.cfg.num_lms, self.max_len)
            tmp_x = torch.mean(torch.cat((x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
            tmp_y = torch.mean(torch.cat((lms_pred_y[i], tmp_nb_y), dim=1), dim=1).view(-1, 1)
            landmark_pred = torch.cat((tmp_x, tmp_y), dim=1)
            landmark_batch.append(landmark_pred)
        landmark_batch = torch.stack([x for x in landmark_batch])
        return landmark_batch


