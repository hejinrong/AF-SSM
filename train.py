import os

import torch
from torch import nn, optim
from torchinfo import summary
from tqdm import tqdm

from dataset.Mydataset import ImageDataset
from model import Resnet, Vgg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model.MedMamba import VSSM
from model.MsMamba import MSVSSM
from model.cross_mamba import MSMamba
from model.cross_vit import CrossViT
from model.deep_vit import DeepViT
from model.simple_vit import SimpleViT

import torchvision
from torchvision import datasets, transforms

import torch
import numpy as np
import random

# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)          # 为CPU设置随机种子
    torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子，如果使用多GPU
    np.random.seed(seed)             # NumPy随机种子
    random.seed(seed)                # Python内置random模块种子
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# 示例
setup_seed(42)  # 42可以替换为你选择的任何整数种子



# 创建数据集

# 划分训练集和测试集
train_dataset = ImageDataset(is_train=True)
test_dataset = ImageDataset(is_train=False)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
# print(len(train_dataset), len(test_dataset))

# for i,j in train_loader:
#     print(i.shape,j.shape)
# print(("==========================================="))
# for i, j in test_loader:
#     print(i.shape, j.shape)
# print(ss[1])
##############################################################################################
n_classes = 2
task = 'binary-class'
model_name = 'vgg19'
w_fold='old02/'
NUM_EPOCHS =100

lr = 0.00003

import torch
from model.vit import ViT
# 获取计算硬件
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = ViT(
#     image_size = 512,
#     patch_size = 32,
#     num_classes = 2,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )
# model =  CrossViT(
#     image_size = 512,
#     num_classes = 2,
#     depth = 4,               # number of multi-scale encoding blocks
#     sm_dim = 192,            # high res dimension
#     sm_patch_size = 16,      # high res patch size (should be smaller than lg_patch_size)
#     sm_enc_depth = 2,        # high res depth
#     sm_enc_heads = 4,        # high res heads
#     sm_enc_mlp_dim = 2048,   # high res feedforward dimension
#     lg_dim = 384,            # low res dimension
#     lg_patch_size = 64,      # low res patch size
#     lg_enc_depth = 3,        # low res depth
#     lg_enc_heads = 4,        # low res heads
#     lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
#     cross_attn_depth = 2,    # cross attention rounds
#     cross_attn_heads = 4,    # cross attention heads
#     dropout = 0.1,
#     emb_dropout = 0.1
# )
# model = SimpleViT(
#     image_size = 512,
#     patch_size = 64,
#     num_classes = n_classes,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048
# )
# model = DeepViT(
#     image_size = 512,
#     patch_size = 64,
#     num_classes = 2,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )
# model = VSSM(num_classes=2)
# model = MSVSSM(num_classes=2)

# model = Resnet.get_resnet34(n_class=n_classes)
model = Vgg.get_vgg19_bn(n_class=n_classes)
# model = MSMamba(
#         image_size=512,
#         num_classes=2,
#         depth=1,  # number of multi-scale encoding blocks
#         sm_dim=192,  # high res dimension
#         sm_patch_size=16,  # high res patch size (should be smaller than lg_patch_size)
#         sm_enc_depth=2,  # high res depth
#         sm_enc_heads=8,  # high res heads
#         sm_enc_mlp_dim=2048,  # high res feedforward dimension
#         lg_dim=384,  # low res dimension
#         lg_patch_size=64,  # low res patch size
#         lg_enc_depth=3,  # low res depth
#         lg_enc_heads=8,  # low res heads
#         lg_enc_mlp_dim=2048,  # low res feedforward dimensions
#         cross_attn_depth=2,  # cross attention rounds
#         cross_attn_heads=8,  # cross attention heads
#         dropout=0.2,
#         emb_dropout=0.2
#     )
# 将模型设置为多GPU训练
model = model.to(device)  # 将模型放到GPU上
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)

#####打印模型参数量
# summary(model, input_size=(32, 3, 512, 512))
# define loss function and optimizer
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=lr)
print(f"task:{task}")

best_test_accuracy = 0


def model_test():
    model.eval()
    y_true = torch.tensor([]).to(device)  # 将输入数据放到GPU上
    y_score = torch.tensor([]).to(device)  # 将目标数据放到GPU上

    data_loader = test_loader
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)  # 将输入数据放到GPU上
            targets = targets.to(device)  # 将目标数据放到GPU上
            outputs = model(inputs)
            # 假设模型输出的是每个类别的概率
            probs = outputs.softmax(dim=-1)
            _, predicted = torch.max(probs, 1)  # 获取最高概率的索引作为预测结果

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, predicted), 0)

    y_true = y_true.cpu().numpy()
    y_score = y_score.cpu().numpy()

    # 计算指标
    accuracy = accuracy_score(y_true, y_score)
    precision = precision_score(y_true, y_score, average='binary')  # 或者使用'binary', 'micro', 'samples', 'weighted'
    recall = recall_score(y_true, y_score, average='binary')
    f1 = f1_score(y_true, y_score, average='binary')

    # 计算特异度
    tn, fp, fn, tp = confusion_matrix(y_true, y_score).ravel()
    specificity = tn / (tn + fp)

    print(f"Accuracy: {accuracy},Precision: {precision},Recall: {recall},F1 Score: {f1}, specificity: {specificity}")
    return accuracy, f1


# train model
for epoch in range(NUM_EPOCHS):
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    # 训练集指标
    train_y_true = torch.tensor([]).to(device)
    train_y_score = torch.tensor([]).to(device)
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(train_loader):
        inputs = inputs.to(device) # 将输入数据放到GPU上
        targets = targets.to(device) # 将目标数据放到GPU上
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
        # 假设模型输出的是每个类别的概率
        probs = outputs.softmax(dim=-1)
        _, predicted = torch.max(probs, 1)  # 获取最高概率的索引作为预测结果

        loss.backward()
        optimizer.step()

        # 累积训练集的预测结果和真实标签
        train_y_true = torch.cat((train_y_true, targets), 0)
        train_y_score = torch.cat((train_y_score, predicted), 0)
        running_loss += loss.item()

    # 计算训练集指标
    train_y_true = train_y_true.cpu().numpy()
    train_y_score = train_y_score.cpu().detach().numpy()

    train_accuracy = accuracy_score(train_y_true, train_y_score)
    train_precision = precision_score(train_y_true, train_y_score, average='binary')
    train_recall = recall_score(train_y_true, train_y_score, average='binary')
    train_f1 = f1_score(train_y_true, train_y_score, average='binary')
    print(
        f"Epoch {epoch + 1}, "
        f"train_loss:{running_loss / len(train_loader)},"
        f"Train Accuracy: {train_accuracy}, "
        f"Train Precision: {train_precision}, "
        f"Train Recall: {train_recall}, "
        f"Train F1: {train_f1}"
    )

    test_accuracy, test_f1 = model_test()
    # 保存最新的最佳模型文件
    if test_accuracy > best_test_accuracy:
        # 删除旧的最佳模型文件(如有)
        old_best_checkpoint_path = 'weight/{}{}-{}best-{:.3f}.pth'.format(w_fold,task, model_name, best_test_accuracy)
        if os.path.exists(old_best_checkpoint_path):
            os.remove(old_best_checkpoint_path)
        # 保存新的最佳模型文件
        best_test_accuracy = test_accuracy
        new_best_checkpoint_path = 'weight/{}{}-{}best-{:.3f}.pth'.format(w_fold,task, model_name, test_accuracy)
        torch.save(model, new_best_checkpoint_path)
        print('保存新的最佳模型', 'weight/{}{}-{}best-{:.3f}.pth'.format(w_fold,task, model_name, best_test_accuracy))

