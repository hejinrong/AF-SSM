import os
import glob
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from dataset.Mydataset import ImageDataset
test_dataset = ImageDataset(is_train=False)
# 创建数据加载器
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
# 推理函数
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
#
# model_SimpleViT = torch.load('weight/old02/binary-class-SimpleViTbest-0.813.pth')
# model_Cross_Vitbest = torch.load('weight/old02/binary-class-CrossViTbest-0.853.pth')
# model_DeepViT = torch.load('weight/old02/binary-class-DeepViTbest-0.773.pth')
# model_VSSM = torch.load('weight/old02/binary-class-VSSMbest-0.933.pth')
# # model_MSMamba = torch.load('weight/old/binary-class-MSMambabest-0.867.pth')
# model_Resnet34 = torch.load('weight/old02/binary-class-resnet34best-0.907.pth')
# model_Vit = torch.load('weight/old02/binary-class-ViTbest-0.840.pth')
# model_Vgg19_bn = torch.load('weight/old02/binary-class-vgg19best-0.880.pth')

model_paths = glob.glob("weight/train7/*.pth")

print(model_paths)


from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC

# 初始化评估指标

# accuracy = Accuracy(task='binary', num_classes=2).cuda()
# precision = Precision(task='binary', average='weighted', num_classes=2).cuda()
# recall = Recall(task='binary', average='weighted', num_classes=2).cuda()
# f1_score = F1Score(task='binary', average='weighted', num_classes=2).cuda()
# auroc = AUROC(task='binary', num_classes=2, average='weighted').cuda()


def model_test(model):
    model.eval()
    y_true = torch.tensor([]).cuda()  # 将输入数据放到GPU上
    y_score = torch.tensor([]).cuda()  # 将目标数据放到GPU上
    test_accuracy = []
    test_precision = []
    test_recall = []
    test_f1_score = []
    test_auroc = []
    data_loader = test_loader
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.cuda()  # 将输入数据放到GPU上
            targets = targets.cuda()  # 将目标数据放到GPU上
            outputs = model(inputs)
            # 假设模型输出的是每个类别的概率
            probs = outputs.softmax(dim=-1)
            _, predicted = torch.max(probs, 1)  # 获取最高概率的索引作为预测结果

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, predicted), 0)

            # accuracy.update(predicted, targets)
            # precision.update(predicted, targets)
            # recall.update(predicted, targets)
            # f1_score.update(predicted, targets)
            # auroc.update(predicted, targets)
            # # 计算最终得分
            # test_accuracy.append(accuracy.compute().cpu())
            # test_precision.append(precision.compute().cpu())
            # test_recall.append(recall.compute().cpu())
            # test_f1_score.append(f1_score.compute().cpu())
            # test_auroc.append(auroc.compute().cpu())

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

    # print(
    #     f"Test Accuracy: {np.mean(test_accuracy)} "
    #     f"Test Precision: {np.mean(test_precision)} "
    #     f"Test Recall: {np.mean(test_recall)} "
    #     f"Test F1 Score: {np.mean(test_f1_score)}"
    #     f"Test AUC: {np.mean(test_auroc)}")
    print(f"Accuracy: {accuracy},Precision: {precision},Recall: {recall},F1 Score: {f1},specificity: {specificity}")
    return accuracy, f1

for i in model_paths:
    print(i)
    _,_ = model_test(torch.load(i))

# print('model_SimpleViT')
# _,_ = model_test(model_SimpleViT)
# print('model_Cross_Vitbest')
# _,_ = model_test(model_Cross_Vitbest)
# print('model_DeepViT')
# _,_ = model_test(model_DeepViT)
# print('model_VSSM')
# _,_ = model_test(model_VSSM)
# print('model_Resnet34')
# _,_ = model_test(model_Resnet34)
# print('model_Vit')
# _,_ = model_test(model_Vit)
# print('model_Vgg19_bn')
# _,_ = model_test(model_Vgg19_bn)
