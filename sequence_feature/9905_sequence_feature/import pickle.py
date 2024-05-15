import pickle
import torch
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from sklearn.decomposition import PCA
import umap




def load_pickle_file(file_path):
    """
    读取并返回指定路径的 pickle 文件中的数据。

    参数:
        file_path (str): pickle 文件的路径。

    返回:
        data (any): 从 pickle 文件中加载的数据。
    """
    # 打开并读取 pickle 文件
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    



def read_csv(file):
    """
    读取并返回指定路径的 CSV 文件中的数据。

    参数:
        file (str): CSV 文件的路径。

    返回:
        df (pandas.DataFrame): 从 CSV 文件中加载的数据。
    """
    try:
        df = pd.read_csv(file, index_col=False, header = None)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

csv_file = 'C://backup//2024//BERT-DGI//graph_feature//9905_pair.csv'
df = read_csv(csv_file)




edges = pd.concat([df[0], df[1]])  # Concatenate the two columns

counter = Counter(edges)  # Count the occurrence of each node

def ranking_edges(counter, n=10):
    sets = []
    for most_common_node, _ in counter.most_common(n):
        target_node = most_common_node
        connected_nodes = set()
        for _, row in df.iterrows():
            if row[0] == target_node:
                connected_nodes.add(row[1])
            elif row[1] == target_node:
                connected_nodes.add(row[0])
        sets.append(connected_nodes)
    return sets

cmap = plt.get_cmap('tab10')

# Generate a list of colors
colors = [cmap(i) for i in range(5)]




sets = ranking_edges(counter)
most_common_node, most_common_count = counter.most_common(1)[0]  # Find the most common node






# print(f"The nodes connected to {target_node} are {connected_nodes}")

# 假设您的张量是这样的:
# tensor 是一个形状为 [1000, 500] 的张量
filepath = 'C://backup//2024//BERT-DGI//sequence_feature//9905_sequence_feature//merged_9905_circRNA_3.pkl'
tensor = load_pickle_file(filepath)
tensor.columns = [str(i) for i in range(len(tensor.columns))]


# 将 PyTorch 张量转换为 NumPy 数组，t-SNE在scikit-learn中基于NumPy实现





# 初始化 t-SNE，设置降维后的维数为2

data = tensor.iloc[:, 1:-1].values
row_labels = tensor.iloc[:, 0].tolist()
# 运行 t-SNE 算法进行降维
# pca = PCA(n_components=2)


# reduced_data = pca.fit_transform(data)

reducer = umap.UMAP()
reduced_data = reducer.fit_transform(data)

# sets_indices = [{row_labels.index(label) for label in s} for s in sets]

indices = [3, 4, 7, 8, 9]

selected_sets = [sets[i] for i in indices]

# Convert the sets of labels to sets of indices
selected_sets_indices = [{row_labels.index(label) for label in s} for s in selected_sets]

# Plot the nodes
for i, (s, color) in enumerate(zip(selected_sets_indices, colors)):
    # Get the points in this set
    set_points = reduced_data[list(s)]
    
    # Plot the points with a label
    plt.scatter(set_points[:, 0], set_points[:, 1], c=color, alpha=0.5, label=f'Set {i+1}')

# Display the legend
plt.legend()

plt.show()