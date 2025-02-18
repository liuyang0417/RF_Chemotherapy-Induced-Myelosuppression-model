import streamlit as st  # 导入 Streamlit 库，用于创建 Web 应用
import pandas as pd  # 导入 Pandas 库，用于数据处理
import pickle  # 导入 pickle 库，用于加载已训练的模型
import os  # 导入 os 库，用于处理文件路径
import shap  # 导入 SHAP 库，用于解释模型
import numpy as np  # 导入 numpy
import matplotlib.pyplot as plt
import io  # 用于将 Matplotlib 图保存到内存缓冲区
# 加载模型# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 组合当前目录与模型文件名，生成模型的完整路径
model_path = os.path.join(current_dir, 'rf_model.pkl')
# 打开并加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)# 使用 pickle 加载模型文件
# 设置 Streamlit 应用的标题
st.title("乳腺癌化疗致骨髓抑制风险预测模型")
# 添加小标题
st.subheader("注：输入特征解释")
st.markdown("""
- **BMI** : 输入**0**为**＜18.5**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;输入 **1** 为 **18.5-24** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;输入 **2** 为 **≥24**
- **合并基础疾病** : 输入 **0** 为  **无**  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;输入 **1** 为 **高血压** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;输入 **2** 为 **糖尿病**
- **化疗方案** : 输入 **0** 为 **AC化疗方案**  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;输入 **1** 为 **AC-T化疗方案** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;输入 **2** 为 **AC-THP化疗方案**
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            输入 **3** 为 **AT化疗方案**  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;输入 **4** 为 **TAC化疗方案** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;输入 **5** 为 **TCbHP化疗方案**
- **是否预防使用集落刺激因子** : 输入 **0** 为 **否**  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;输入 **1**   为 **是**
- **乳腺癌分型** : 输入 **0** 为 **三阴性乳腺癌**  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;输入 **1**   为 **Her-2阳性**   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;输入 **3** 为 **激素受体阳性**

""")#&nbsp;为空格，使得文字在stramlit上显示的更容易区分
# 在侧边栏中输入特征
st.sidebar.header("输入特征")  # 侧边栏的标题
# 添加化疗周期的解释


# 使用下拉框选择BMI的选项
bmi_option = st.sidebar.selectbox(
    "BMI",
    options=["0", "1", "2"],
    index=0  # 默认选中第一个选项（"<18.5"）
)
st.sidebar.write("您选择的BMI范围为：", bmi_option)

# 使用下拉框选择合并基础疾病的选项
comorbidity_option = st.sidebar.selectbox(
    "合并基础疾病",
    options=["0", "1", "2"],
    index=0  # 默认选中第一个选项（"无"）
)
st.sidebar.write("您选择的合并基础疾病为：", comorbidity_option)

# 使用下拉框选择化疗方案的选项
chemotherapy_regimen_option = st.sidebar.selectbox(
    "化疗方案",
    options=["0", "1", "2", "3", "4", "5"],
    index=0  # 默认选中第一个选项（"AC"）
)
st.sidebar.write("您选择的淋巴细胞数值选项为：", chemotherapy_regimen_option)

# 使用下拉框选择化疗周期的选项
chemotherapy_cycle_option = st.sidebar.selectbox(
    "化疗周期",
    options=["0", "1"],
    index=0  # 默认选中第一个选项（"1-4"）
)
st.sidebar.write("您选择的化疗周期为：", chemotherapy_cycle_option)

# 使用下拉框选择是否预防使用集落刺激因子
use_growth_factor = st.sidebar.selectbox(
    "是否预防使用集落刺激因子",
    options=["0", "1"],
    index=0  # 默认选中第一个选项（"否"）
)
st.sidebar.write("您的选择是：", use_growth_factor)

# 使用下拉框选择乳腺癌分型的选项
breast_cancer_subtype_option = st.sidebar.selectbox(
    "乳腺癌分型",
    options=["0", "1", "2"],
    index=0  # 默认选中第一个选项（"三阴性"）
)
st.sidebar.write("您选择的乳腺癌分型为：", breast_cancer_subtype_option)

# 使用文本框输入白细胞数值
white_blood_cell_count = st.sidebar.number_input(
    "白细胞（x10^9）",  # 修改了提示文本
    min_value=0.0,  # 最小值
    max_value=50.0,  # 最大值
    value=4.0,  # 默认值
    step=0.01  # 步长，每次增减的单位
)
st.sidebar.write("您输入的白细胞数值为：", white_blood_cell_count)

# 使用文本框输入血小板计数
platelet_count = st.sidebar.number_input(
    "血小板计数（x10^9）",  # 修改了提示文本
    min_value=0.0,  # 最小值
    max_value=1000.0,  # 最大值，血小板计数通常更高
    value=250.0,  # 默认值，正常范围一般在100-300之间
    step=1.0  # 步长，每次增减的单位
)
st.sidebar.write("您输入的血小板计数为：", platelet_count)

# 使用文本框输入淋巴细胞数值
serum_lymphocyte_count = st.sidebar.number_input(
    "淋巴细胞数（x10^9）",
    min_value=0.0,  # 最小值
    max_value=20.0,  # 最大值
    value=0.8,  # 默认值
    step=0.01  # 步长，每次增减的单位
)
st.sidebar.write("您输入的淋巴细胞数值为：", serum_lymphocyte_count)

# 使用文本框输入血清白蛋白的值
serum_albumin = st.sidebar.number_input(
    "血清白蛋白（g/L）",
    min_value=0.0,  # 最小值
    max_value=100.0,  # 最大值
    value=34.0,  # 默认值
    step=0.01  # 步长，每次增减的单位
)
st.sidebar.write("您输入的血清白蛋白值为：", serum_albumin)

# 将所有输入特征整理为 DataFrame 格式
input_data = pd.DataFrame({
    "chemotherapy regimens": [chemotherapy_regimen_option],
    "cycle": [chemotherapy_cycle_option],
    "G-CSF": [use_growth_factor],
    "BMI": [bmi_option],
    "underlying disease": [comorbidity_option],
    "breast cancer types": [breast_cancer_subtype_option],
    "ALC": [serum_lymphocyte_count],
    "ALB": [serum_albumin],
    "leukocyte": [white_blood_cell_count],
    "platelet": [platelet_count]
})
# 添加预测按钮，用户点击后进行模型预测
#if st.button("预测"):
#    prediction = model.predict(input_data)  # 使用加载的模型进行预测
#    st.write(f"预测结果: {prediction[0]}")#预测发生的类别
# 添加预测按钮，用户点击后进行模型预测
if st.button("预测"):
    # 获取预测概率
    prediction_prob = model.predict_proba(input_data)
    # 对于二分类，第二列是类别 1 的概率
    probability_class_1 = prediction_prob[0][1]
    st.markdown(f"<h3>发生骨髓抑制的概率为: <span style='color: red;'>{probability_class_1:.3f}</span></h3>",
                unsafe_allow_html=True)
    # 根据概率判断风险级别
    if probability_class_1 < 0.36:
        risk_level = "低风险"
    elif 0.37 <= probability_class_1 <= 0.57:
        risk_level = "中风险"
    else:
        risk_level = "高风险"

    # 显示风险级别
    st.markdown(f"<h3>风险级别: <span style='color: red;'>{risk_level}</span></h3>", unsafe_allow_html=True)

    # 计算 SHAP 值
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    st.subheader("模型预测的瀑布图")
    # 创建一个合适大小的图形
    plt.figure(figsize=(10, 8))  # 设置图形大小为 10x8 英寸

    # 绘制 SHAP 瀑布图
    shap.plots.waterfall(shap_values[0, :, 1])

    # 调整布局，确保不会被截断
    plt.tight_layout()

    # 将 matplotlib 图形保存为 BytesIO 对象，然后在 Streamlit 中显示
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # 将图形显示在 Streamlit 页面
    st.image(buf)
    buf.close()
    st.markdown("""
        注：纵轴表示预测模型中的特征变量，最左侧代表特征变量值；横轴表示 SHAP 值，红色表示特征变量对预测结果的正向贡献（增加骨髓抑制的风险），反之亦然。
        E[f(x)] 为基线值，表示当模型不考虑特征变量影响时，模型的平均预测值；f(x) 为预测值，表示模型在考虑特征变量影响后，最终的预测值。</p>
        """, unsafe_allow_html=True)#<p>是换行符号