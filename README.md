<div align="center">
<h1>LSTM Baseline</h1>
A LSTM model baseline implementation based Pytorch
<br>
<br>
  
**English**| [**中文繁體**](README_CN.md) |
</div>


## About
This is a simple implementation of LSTM model by using pytorch lightning for classifying MNIST dataset.

Using this code could be a good start to explore the power of LSTM in sequence data.
## Course Requirements
According to Davison Wang speech, you should implement your LSTM code with the following requirements:

- Use **MNIST** to be your **datasets**and do a **classification task**.
- Use **LSTM** to be part of the network architecture
- Use **F1 Score** to be your scoring scheme
- Ensure your score **better than** this baseline(**0.98 in F1 Score**)
- Provide your **model weight file** (usually ending in pth or weights format) in the project's README for reproduction.
## Framework Choice
You could use **YOLO** or **Pytorch** or any other choice if you want. There is no requirement to follow the same framework as this baseline.

## Requirements
- conda
## Installation
```bash
git clone https://github.com/kaze-desu/LSTM-baseline-BADE03.git
cd LSTM-baseline-BADE03
conda env create -f environment.yml -n lstm
conda activate lstm
```
## Manual Installation
```bash
conda create -n lstm python=3.11
conda activate lstm
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install pytorch-lightning
pip install matplotlib
pip install numpy
pip install scikit-learn
```
## Baseline
| Model | Loss | F1 Score |
|-------|------|----------|
| [LSTM](https://studentmust-my.sharepoint.com/:u:/g/personal/1220026920_student_must_edu_mo/EcdJZfFRcLtKmeNzTLwOjEwBh8uUJlxxqtGHTzlXxPMynw?e=EYJQQq)  | 0.07 | 0.98     |
## How to duplicate baseline
- Clone the project and download the model provided above, then follow the **Test** part.
## Train
You can train the model by running the following command:
```bash
python main.py
```
## Test
```bash
python main.py --eval <path_to_model>
```
## Encountering Issues with This Project's Code?

Please use the "issue" feature to submit your problem. Make sure your issue includes the following details:

- Environment information  
- Cause of the issue  
- Expected outcome  
- Steps to reproduce  

Additionally, if you are able to resolve the issue, you are encouraged to submit a pull request (PR) to the repository and merge your solution, contributing to the improvement of this repository.
## Want A Better Results?
### Simple Hyperparameter Tuning
You can try the following:
- Increase the number of epochs
- Change the model architecture
- Change the learning rate
### Use better architectures
**Davison Wang** recommends you to read the following paper:
- https://www.sciencedirect.com/science/article/pii/S0893608021003439
## References
https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/#steps_3

## Acknowledgement
- Davison Wang
