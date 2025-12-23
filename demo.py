import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# 页面标题
st.set_page_config(page_title="ID3 决策树实验报告", layout="wide")
st.title("ID3 算法多数据集实验报告")
st.markdown("该面板展示了 ID3 算法在 4 个 UCI 数据集上的表现、指标统计及决策树可视化。")

def load_uci_data(name):
    if name == 'iris': return datasets.load_iris()
    elif name == 'wine': return datasets.load_wine()
    elif name == 'cancer': return datasets.load_breast_cancer()
    elif name == 'digits': return datasets.load_digits()
    return None

def run_experiment():
    dataset_names = ['iris', 'wine', 'cancer', 'digits']
    results_list = []
    
    # 创建两列布局：左侧显示表格，右侧显示图表
    col1, col2 = st.columns([1, 1])

    for name in dataset_names:
        data = load_uci_data(name)
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        depth = clf.get_depth()
        
        results_list.append({
            'Dataset': name,
            'Train Size': len(X_train),
            'Test Size': len(X_test),
            'Accuracy': acc,
            'Tree Depth': depth
        })

        # --- 在 Streamlit 中展示每棵树 ---
        with st.expander(f"查看 {name} 数据集的决策树结构 (Accuracy: {acc:.2%})"):
            fig, ax = plt.subplots(figsize=(20, 10))
            # 仅展示前3层，保持美观
            plot_tree(clf, filled=True, feature_names=data.feature_names, ax=ax, max_depth=3, fontsize=10)
            st.pyplot(fig)

    # --- 汇总结果展示 ---
    results_df = pd.DataFrame(results_list)
    
    # 1. 展示数据表格
    st.subheader("📊 实验结果汇总表")
    st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy'], color='#90EE90'))

    # 2. 展示对比图表
    st.subheader("📈 准确率与深度对比")
    
    tab1, tab2 = st.tabs(["准确率 (Accuracy)", "树深度 (Tree Depth)"])
    with tab1:
        # labelAngle=0 强制 X 轴标签横向显示
        chart = alt.Chart(results_df).mark_bar().encode(
            x=alt.X('Dataset', axis=alt.Axis(labelAngle=0)),
            y='Accuracy',
            tooltip=['Dataset', 'Accuracy']
        )
        st.altair_chart(chart, use_container_width=True)

    with tab2:
        chart = alt.Chart(results_df).mark_bar().encode(
            x=alt.X('Dataset', axis=alt.Axis(labelAngle=0)),
            y='Tree Depth',
            tooltip=['Dataset', 'Tree Depth']
        )
        st.altair_chart(chart, use_container_width=True)

# 运行主函数
if __name__ == "__main__":
    run_experiment()