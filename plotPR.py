import torch
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_recall_curve, roc_curve, auc
# from torch.utils.data import DataLoader
from dataset.Mydataset import ImageDataset

testdataset = ImageDataset(is_train=False)
# 创建数据加载器
testloader = torch.utils.data.DataLoader(testdataset, batch_size=16, shuffle=False)

#
# # 加载模型的函数
# def load_model(model_path):
#     model = torch.load(model_path)  # 确保你的模型定义与训练时相同
#     model.eval()  # 设置为评估模式
#     return model
#
#
# # 计算模型预测概率的函数
# def get_model_predictions(model, testloader):
#     y_true = torch.tensor([]).cuda()  # 将输入数据放到GPU上
#     y_score = torch.tensor([]).cuda()  # 将目标数据放到GPU上
#     with torch.no_grad():  # 不计算梯度，减少内存消耗
#         for inputs, labels in testloader:
#             inputs = inputs.cuda()  # 将输入数据放到GPU上
#             labels = labels.cuda()  # 将目标数据放到GPU上
#             outputs = model(inputs)
#             # probabilities = torch.softmax(outputs, dim=1)[:, 1]  # 假设1是正类的索引
#             probs = outputs.softmax(dim=-1)
#             _, predicted = torch.max(probs, 1)  # 获取最高概率的索引作为预测结果
#
#             y_true = torch.cat((y_true, labels), 0)
#             y_score = torch.cat((y_score, predicted), 0)
#         y_true = y_true.cpu().numpy()
#         y_score = y_score.cpu().numpy()
#     return y_true, y_score
#
#
# model_paths = ['weight/old02/binary-class-SimpleViTbest-0.813.pth',
#                'weight/old02/binary-class-CrossViTbest-0.853.pth',
#                'weight/old02/binary-class-DeepViTbest-0.773.pth',
#                'weight/old02/binary-class-VSSMbest-0.933.pth',
#                'weight/old/binary-class-MSMambabest-0.867.pth',
#                'weight/old02/binary-class-resnet34best-0.907.pth',
#                'weight/old02/binary-class-ViTbest-0.840.pth',
#                'weight/old02/binary-class-vgg19best-0.880.pth']
#
# plt.figure(figsize=(12, 6))
#
# # 绘制PR曲线
# plt.subplot(1, 2, 1)
# for path in model_paths:
#     model = load_model(path)
#     y_true, y_scores = get_model_predictions(model, testloader)
#     precision, recall, _ = precision_recall_curve(y_true, y_scores)
#     plt.plot(recall, precision, lw=2, label=f'{path}')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend()
#
# # 绘制ROC曲线
# plt.subplot(1, 2, 2)
# for path in model_paths:
#     model = load_model(path)
#     y_true, y_scores = get_model_predictions(model, testloader)
#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, lw=2, label=f'{path} (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend()
#
# plt.tight_layout()
#
# plt.savefig('各类别PR曲线.pdf', bbox_inches='tight')
# plt.show()


import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from torch.utils.data import DataLoader
# 假设你的testloader是这样定义的
# testloader = DataLoader(your_test_dataset, batch_size=batch_size, shuffle=False)

# 加载模型的函数
def load_model(model_path):
    model = torch.load(model_path)  # 确保你的模型定义与训练时相同
    model.eval()  # 设置为评估模式
    return model

# 计算模型预测概率的函数
def get_model_predictions(model, testloader):
    y_true = []
    y_scores = []
    with torch.no_grad():  # 不计算梯度，减少内存消耗
        for inputs, labels in testloader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # 假设1是正类的索引
            y_scores.extend(probabilities.tolist())
            y_true.extend(labels.tolist())
    return y_true, y_scores

model_paths = ['weight/train7/binary-class-ViTbest-0.697.pth',
 'weight/train7/binary-class-MSVSSMbest-0.866.pth',
 'weight/train7/binary-class-VSSMbest-0.857.pth',
 'weight/train7/binary-class-CrossViTbest-0.571.pth',
 'weight/train7/binary-class-SimpleViTbest-0.697.pth',
 'weight/train7/binary-class-DeepViTbest-0.647.pth',
 'weight/train7/binary-class-resnet34best-0.857.pth',
 'weight/train7/binary-class-vgg19_bnbest-0.840.pth']


model_names =  ['Vit','AF-SSM','MedMamba','CrossVit','SimpleVit','DeepVit','Resnet34','Vgg19']

plt.figure(figsize=(12, 6))

# 绘制PR曲线
plt.subplot(1, 2, 1)
for path,name in zip(model_paths,model_names):
    model = load_model(path)
    y_true, y_scores = get_model_predictions(model, testloader)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision, lw=2, label=f'{name}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

# 绘制ROC曲线
plt.subplot(1, 2, 2)
for path,name in zip(model_paths,model_names):
    model = load_model(path)
    y_true, y_scores = get_model_predictions(model, testloader)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()

plt.tight_layout()
plt.savefig('PR.pdf',dpi=1200, bbox_inches='tight')
plt.show()


import seaborn as sns
from sklearn.metrics import confusion_matrix

# 加载模型的函数
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

# 获取模型预测的函数
def get_predictions(model, testloader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in testloader:

            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds


# 所有子图都绘制在一行上
fig, axs = plt.subplots(1, len(model_paths), figsize=(20, 5))  # 调整子图布局

for idx, (path, name) in enumerate(zip(model_paths, model_names)):
    model = load_model(path)
    y_true, y_pred = get_predictions(model, testloader)
    conf_mat = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=axs[idx],cbar=False)
    axs[idx].set_aspect('equal')
    axs[idx].set_title(name)
    axs[idx].set_xlabel('Predicted labels')
    axs[idx].set_ylabel('True labels')
    axs[idx].set_aspect('equal', 'box')



plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('Metrics.pdf',dpi=1200, bbox_inches='tight')
plt.show()