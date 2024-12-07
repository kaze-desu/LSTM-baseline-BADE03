### About
This is a simple implementation of LSTM model by using pytorch lightning for classifying MNIST dataset.

Use this code could be a good start to explore the power of LSTM in sequence data.

### Requirements
- conda
### Installation
```bash	
conda create -n lstm -f environment.yml
```
### Baseline
| Model | Loss | F1 Score |
|-------|------|----------|
| [LSTM](https://studentmust-my.sharepoint.com/:u:/g/personal/1220026920_student_must_edu_mo/EcdJZfFRcLtKmeNzTLwOjEwBh8uUJlxxqtGHTzlXxPMynw?e=EYJQQq)  | 0.07 | 0.98     |

### Train
You can train the model by running the following command:
```bash
python main.py
```
### Test
```bash
python main.py --eval <path_to_model>
```
### Want Better Results?
#### Simple Hyperparameter Tuning
You can try the following:
- Increase the number of epochs
- Change the model architecture
- Change the learning rate
#### Use better architectures
**Davison Wang** recommands you to read the following paper:
- https://www.sciencedirect.com/science/article/pii/S0893608021003439
### References
https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/#steps_3

### Acknowledgement
- Davison Wang