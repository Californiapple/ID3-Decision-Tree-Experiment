import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# 设置绘图风格
plt.style.use('ggplot')

# 加载数据集函数
def load_uci_data(name):
    if name == 'iris': return datasets.load_iris()
    elif name == 'wine': return datasets.load_wine()
    elif name == 'cancer': return datasets.load_breast_cancer()
    elif name == 'digits': return datasets.load_digits()
    return None

# 主运行函数，生成并保存图片
def run_and_save_images():
    dataset_names = ['iris', 'wine', 'cancer', 'digits']
    results_list = []

    print("正在生成图片，请稍候...")

    for name in dataset_names:
        # 1. 数据加载与处理
        data = load_uci_data(name)
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # 2. 模型训练 (criterion='entropy' 模拟 ID3)
        clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        depth = clf.get_depth()
        
        results_list.append({
            'Dataset': name,
            'Accuracy': acc,
            'Tree Depth': depth
        })

        # 3. 保存决策树图片
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(clf, filled=True, feature_names=data.feature_names, 
                  ax=ax, max_depth=3, fontsize=10)
        plt.title(f"Decision Tree for {name} (Acc: {acc:.2%})")
        # 保存为 png 文件
        plt.savefig(f"tree_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: tree_{name}.png")

    # 4. 生成对比图表
    results_df = pd.DataFrame(results_list)
    
    # 准确率对比图
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Dataset', y='Accuracy', data=results_df, palette='viridis')
    plt.ylim(0, 1.1)
    plt.title("Accuracy Comparison")
    for index, row in results_df.iterrows():
        plt.text(index, row.Accuracy + 0.02, f'{row.Accuracy:.2f}', ha='center')
    plt.savefig("acc_comparison.png", dpi=300)
    plt.close()

    # 树深度对比图
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Dataset', y='Tree Depth', data=results_df, palette='magma')
    plt.title("Tree Depth Comparison")
    for index, row in results_df.iterrows():
        plt.text(index, row['Tree Depth'] + 0.2, f'{row["Tree Depth"]}', ha='center')
    plt.savefig("depth_comparison.png", dpi=300)
    plt.close()
    
    print("\n所有图片生成完毕！")
    print("-" * 30)
    print("LaTeX 表格数据如下 (可直接复制到 LaTeX 表格中):")
    for res in results_list:
        print(f"{res['Dataset'].capitalize()} & {res['Accuracy']:.4f} & {res['Tree Depth']} \\\\ \\hline")

if __name__ == "__main__":
    run_and_save_images()