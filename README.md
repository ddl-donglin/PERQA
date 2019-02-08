## A Conversational Question Answering on Phrase Ellipsis Recovery

### Baseline
Base projects:
[flowQA](https://github.com/momohuang/FlowQA)
[coQA](https://github.com/stanfordnlp/coqa-baselines)
[DrQA](https://github.com/facebookresearch/DrQA)
[QuAC](http://quac.ai/) (dataset)

Modified projects:
[flowQA_4_PERQA](https://github.com/Daviddddl/FlowQA)
[coQA_4_PERQA](https://github.com/Daviddddl/FlowQA/blob/master/QA_model/model_CoQA.py)
[QuAC_4_PERQA](https://github.com/Daviddddl/FlowQA/blob/master/QA_model/model_QuAC.py)
[DrQA_4_PERQA](https://github.com/Daviddddl/DrQA)


### Data

Download the PERQA dataset: [PERQA](https://drive.google.com/open?id=1_KP3YOeCrpwuV8Qecq2RXyBvPkwso-7-)


### Environment

Tensorflow1.8 + CUDA9.0 + python3.6 + Tensor2tensor

  
  download and install Anaconda 
  https://www.anaconda.com/download/#linux
  
  for more install details: 
  https://blog.csdn.net/Davidddl/article/details/81873606
  
```
sudo bash Anaconda3-5.2.0-Linux-x86_64.sh

conda --version

conda create -n tensorflow pip python=3.6

source activate tensorflow

pip install --upgrade pip

(tensorflow)$ pip install --ignore-installed --upgrade https://download.tensorflow.google.cn/linux/gpu/tensorflow_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl
```


### Encoding

#### Bert
##### Download pre-trained encoder model: [BERT_4_Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) 
If you want to extract your own proposal data, you could just run the utils/prepare.py (some paths of files need to be modified:
```shell
source activate tensorflow
pip install -r requirements.txt
python prepare.py
```
##### Download extracted bert features of [PERQA](https://drive.google.com/open?id=1_KP3YOeCrpwuV8Qecq2RXyBvPkwso-7-)
[PERQA_p1](https://drive.google.com/open?id=1yP0mbo2DI7X-CujPpAJ_TWnxM6niOqBY)
[PERQA_p2](https://drive.google.com/open?id=1yP0mbo2DI7X-CujPpAJ_TWnxM6niOqBY)


#### GloVe
modified glove for Chinese. [glove-tools](https://github.com/Daviddddl/glove-tools)
If you want to extract your own proposal data, you could find more details from the modified [glove-tools](https://github.com/Daviddddl/glove-tools)
Due to lack of support for Chinese in [basic projects](https://github.com/maciejkula/glove-python), we adopted [LTP](https://github.com/HIT-SCIR/ltp), [LTP_py](https://github.com/HIT-SCIR/pyltp)
You will find the usage in [glove-tools](https://github.com/Daviddddl/glove-tools)

#### CoVE
[cove](https://github.com/salesforce/cove)


#### ELMo
Bidirectional LSTM-CRF and ELMo
[anago](https://github.com/Hironsan/anago)
