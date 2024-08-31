import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from collections import Counter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim.downloader as api
import os
from functools import lru_cache

# 加载预训练的词向量模型
word2vec_model = api.load("glove-wiki-gigaword-50")

# 用于缓存已生成的词向量
word_vector_cache = {}


# 定义一个函数，根据词在 Word2Vec 模型中是否存在来生成词向量
@lru_cache(maxsize=None)
def get_word_vector(word, model):
    global word_vector_cache

    if word in word_vector_cache:
        return word_vector_cache[word]

    if word in model:
        vector = model[word]
    else:
        # 对于未登录词，生成一个随机向量
        vector = np.random.rand(model.vector_size)

    # 将生成的向量缓存起来
    word_vector_cache[word] = vector
    return vector


def get_commands_from_folder(folder_path):
    all_commands = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip().split()
                    all_commands.append(line)
    return all_commands


def build_feature_vector(command, word_counter, model, label_mapping, max_len=10):
    word_vectors = np.array([get_word_vector(word, model) for word in command])
    word_freqs = np.array([word_counter[word] for word in command])

    if len(word_freqs) < max_len:
        word_freqs = np.pad(word_freqs, (0, max_len - len(word_freqs)), 'constant')
    else:
        word_freqs = word_freqs[:max_len]

    mean_pool = np.mean(word_vectors, axis=0)
    max_pool = np.max(word_vectors, axis=0)
    min_pool = np.min(word_vectors, axis=0)

    command_length = len(command)
    freq_1_count = np.sum(word_freqs == 1)
    freq_2_plus_count = np.sum(word_freqs > 2)

    command_type = 0  # 默认类型
    for word in command:
        if word in label_mapping:
            command_type = label_mapping[word]
            break

    freq_1_count_weighted = freq_1_count * 10
    freq_2_plus_count_weighted = freq_2_plus_count * 10

    feature_vector = np.concatenate([min_pool, max_pool, mean_pool,
                                     [command_length, freq_1_count_weighted, freq_2_plus_count_weighted, command_type]])
    return feature_vector


def read_command(path):
    new_command = []
    with open(path, 'r') as file:
        for line in file:
            line = line.strip().split()
            new_command.append(line)
    return new_command


# 评估函数
def evaluate_predictions(true_labels, scores, threshold=0.5):
    predictions = (scores >= threshold).astype(int)
    accuracy = np.mean(predictions == true_labels)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, scores)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'ROC AUC: {roc_auc}')

    conf_matrix = confusion_matrix(true_labels, predictions)
    print('Confusion Matrix:')
    print(conf_matrix)

    fpr, tpr, _ = roc_curve(true_labels, scores)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    return accuracy, precision, recall, f1, roc_auc


def hard_voting(predictions):
    # 计算每个样本的投票结果
    combined_predictions = np.sum(predictions, axis=0) > (len(predictions) / 2)
    return combined_predictions.astype(int)


# 加载训练数据
commands = get_commands_from_folder('dataset/train')
all_words = [word for command in commands for word in command]
word_counter = Counter(all_words)

label_mapping = {"rundll32.exe": 1, "dllhost.exe": 2,"svchost.exe":3 }  # 示例标签映射

features = []
max_len = max(len(command) for command in commands)
for command in commands:
    feature_vector = build_feature_vector(command, word_counter, word2vec_model, label_mapping, max_len)
    features.append(feature_vector)

features = np.array(features)

# 标准化特征数据
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 使用LOF进行异常检测
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_predictions = lof.fit_predict(features_scaled)

# 使用Isolation Forest进行异常检测
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest_predictions = iso_forest.fit_predict(features_scaled)

# 使用One-Class SVM进行异常检测
one_class_svm = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
one_class_svm_predictions = one_class_svm.fit_predict(features_scaled)

# 测试数据加载和预处理
test_commands = read_command('dataset/test/test.txt')
test_features = []
test_labels = []

for command in test_commands:
    feature_vector = build_feature_vector(command[:-1], word_counter, word2vec_model, label_mapping, max_len)
    test_features.append(feature_vector)
    test_labels.append(int(command[-1]))  # 解析手动添加的标签

test_features = np.array(test_features)
test_features_scaled = scaler.transform(test_features)
true_labels = np.array(test_labels)

# 使用LOF进行预测
test_lof_predictions = lof.fit_predict(test_features_scaled)
test_lof_predictions = np.where(test_lof_predictions == -1, 1, 0)

# 使用Isolation Forest进行预测
test_iso_forest_predictions = iso_forest.predict(test_features_scaled)
test_iso_forest_predictions = np.where(test_iso_forest_predictions == -1, 1, 0)

# 使用One-Class SVM进行预测
test_one_class_svm_predictions = one_class_svm.predict(test_features_scaled)
test_one_class_svm_predictions = np.where(test_one_class_svm_predictions == -1, 1, 0)

# 收集所有模型的预测结果
all_predictions = np.array([test_lof_predictions, test_iso_forest_predictions, test_one_class_svm_predictions])

# 应用硬投票
voting_predictions = hard_voting(all_predictions)

# 评估投票结果
print("Voting Results:")
evaluate_predictions(true_labels, voting_predictions)

# 输出每个命令行的预测结果
print("\nIndividual Model Results:")
print("LOF Results:")
evaluate_predictions(true_labels, test_lof_predictions)

print("\nIsolation Forest Results:")
evaluate_predictions(true_labels, test_iso_forest_predictions)

print("\nOne-Class SVM Results:")
evaluate_predictions(true_labels, test_one_class_svm_predictions)

# 输出每个命令行的预测结果
for i, command in enumerate(test_commands):
    print(f"Command: {' '.join(command[:-1])}, True Label: {true_labels[i]}, "
          f"Voting Prediction: {voting_predictions[i]}, "
          f"LOF Prediction: {test_lof_predictions[i]}, "
          f"Isolation Forest Prediction: {test_iso_forest_predictions[i]}, "
          f"One-Class SVM Prediction: {test_one_class_svm_predictions[i]}")

# 可视化（使用PCA降维）
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(test_features_scaled)

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.scatter(reduced_features[test_lof_predictions == 0, 0], reduced_features[test_lof_predictions == 0, 1],
            label='Normal')
plt.scatter(reduced_features[test_lof_predictions == 1, 0], reduced_features[test_lof_predictions == 1, 1],
            label='Anomaly')
plt.title('PCA of Command Features (LOF)')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(reduced_features[test_iso_forest_predictions == 0, 0], reduced_features[test_iso_forest_predictions == 0, 1],
            label='Normal')
plt.scatter(reduced_features[test_iso_forest_predictions == 1, 0], reduced_features[test_iso_forest_predictions == 1, 1],
            label='Anomaly')
plt.title('PCA of Command Features (Isolation Forest)')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(reduced_features[test_one_class_svm_predictions == 0, 0], reduced_features[test_one_class_svm_predictions == 0, 1],
            label='Normal')
plt.scatter(reduced_features[test_one_class_svm_predictions == 1, 0], reduced_features[test_one_class_svm_predictions == 1, 1],
            label='Anomaly')
plt.title('PCA of Command Features (One-Class SVM)')
plt.legend()

plt.tight_layout()
plt.show()
