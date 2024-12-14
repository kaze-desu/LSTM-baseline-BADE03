<div align="center">
<h1>LSTM Baseline</h1>
基於pytorch的LSTM基準代碼實現
<br>
<br>
  
[**English**](README.md) | **中文繁體** |
</div>

## 關於
本項目是一個利用LSTM模型對MNIST數據集進行分類任務的簡單實現，程序基於pytorch lightning深度學習框架。

你可以使用本代碼作爲探索LSTM模型的起點。
## BADE03課程需求
据Davison Wang老師所説，你應當確保在自己的LSTM項目中實現以下功能：

- 使用**MNIST**數據集進行分類任務
- 使用**LSTM**模型作爲你的主要網絡結構
- 利用**F1 Score**進行分類任務的評分
- 確保你的項目最終評估分數高於本基準值(**在F1 Score下達到98%的準確度**)
- 在項目的README中提供你的**模型權重文件**（通常以pth或weights格式結尾），以便復現。
## 深度學習框架選擇
你可以選擇使用課上講的**YOLO**作爲實現代碼的框架，也可以跟本項目一樣，選擇**Pytorch**進行實現。無論選擇哪種都是可以的，隨著自己喜好來吧。

## 環境需求
- conda/mamba
- pytorch==2.4.0
- pytorch-lightning==2.4.0
- lightning==2.4.0
- matplotlib
- numpy
- scikit-learn
## 安裝
```bash
git clone https://github.com/kaze-desu/LSTM-baseline-BADE03.git
cd LSTM-baseline-BADE03
conda env create -f environment.yml -n lstm
conda activate lstm
```
## 手動安裝
```bash
conda create -n lstm python=3.11
conda activate lstm
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install pytorch-lightning
pip install lightning
pip install matplotlib
pip install numpy
pip install scikit-learn
```
## 基準值
| Model | Loss | F1 Score |
|-------|------|----------|
| [LSTM](https://studentmust-my.sharepoint.com/:u:/g/personal/1220026920_student_must_edu_mo/EcdJZfFRcLtKmeNzTLwOjEwBh8uUJlxxqtGHTzlXxPMynw?e=EYJQQq)  | 0.07 | 0.98     |
## 如何復現
- 將本項目克隆到本地，並下載上方提供的模型，隨後執行**測試**篇章的命令即可。
## 訓練
你可以修改項目代碼后，對模型進行進一步的訓練微調：
```bash
python main.py
```
## 測試
```bash
python main.py --eval <path_to_model>
```
## 遇到本項目代碼問題？
請使用issue功能提交你的問題，請注意你的issue當中應該包含：
- 環境信息
- 問題起因
- 預期效果
- 復現步驟

此外，如果有能力，你可以將問題解決后利用PR功能提交至倉庫並合并，幫助完善本倉庫。
## 想要超過基準值？
### 試試簡單的調整超參數參吧
你可以根據以下步驟優化模型代碼

- 增加訓練迭代次數
- 改變模型網絡架構
- 改變學習率
### 使用更好的網絡結構
**Davison Wang** 推薦你們參閲這篇優化論文:
- https://www.sciencedirect.com/science/article/pii/S0893608021003439
## 引用
https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/#steps_3

## 鳴謝
- Davison Wang
