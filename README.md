<!-- ssh dldi@next-gpu1.d2.comp.nus.edu.sg -->

## A Conversational Question Answering on Phrase Ellipsis Recovery

### Baseline
If you want to train your own data, please follow the 
[glove](https://github.com/Daviddddl/glove-tools.git) and [spacy](https://github.com/Daviddddl/spacy-zh.git), 
and generate your own features.

|Base projects|[flowQA](https://github.com/momohuang/FlowQA)|[coQA](https://github.com/stanfordnlp/coqa-baselines)|[DrQA](https://github.com/facebookresearch/DrQA)|[QuAC](http://quac.ai/) (dataset)|
|---|---|---|---|---|

|Modified projects|[flowQA_4_PERQA](https://github.com/Daviddddl/FlowQA)|[coQA_4_PERQA](https://github.com/Daviddddl/FlowQA/blob/master/QA_model/model_CoQA.py)|[QuAC_4_PERQA](https://github.com/Daviddddl/FlowQA/blob/master/QA_model/model_QuAC.py)|[DrQA_4_PERQA](https://github.com/Daviddddl/DrQA)|
|---|---|---|---|---|

<!--

### Data

Download the PERQA dataset: [PERQA](https://drive.google.com/open?id=1_KP3YOeCrpwuV8Qecq2RXyBvPkwso-7-)

-->

#### QuAC
parameters:
```shell
gpu5 quac_perq0:
    CoVe_opt=1, 
    MTLSTM_path='glove/MT-LSTM.pth', 
    answer_type_num=1, 
    batch_size=3, 
    concat_rnn=False, 
    cuda=True, 
    deep_att_hidden_size_per_abstr=250, 
    deep_inter_att_do_similar=0, 
    dev_dir='QuAC_data/', 
    do_hierarchical_query=True, 
    do_not_save=False, 
    do_prealign=True, 
    do_ptr_update=True, 
    do_residual=False, 
    do_residual_everything=False, 
    do_residual_rnn=False, 
    do_seq_dropout=True, 
    dropout_emb=0.4, 
    elmo_batch_size=12, 
    elmo_lambda=0.0, 
    epoches=30, 
    eval_per_epoch=1, 
    explicit_dialog_ctx=2, 
    final_output_att_hidden=250, 
    fix_embeddings=False, 
    grad_clipping=10, 
    hidden_size=125, 
    learning_rate=0.1, 
    log_file='output.log', 
    log_per_updates=20, 
    max_len=35, 
    model_dir='models', 
    momentum=0, 
    my_dropout_p=0.4, 
    name='', 
    ner_dim=8, 
    ner_size=19, 
    no_dialog_flow=False, 
    no_em=False, 
    optimizer='adamax', 
    pos_dim=12, 
    pos_size=51, 
    prealign_hidden=300, 
    prealign_option=2, 
    pretrain='', 
    ptr_net_attn_type='Bilinear', 
    ptr_net_indep_attn=True, 
    question_merge='linear_self_attn', 
    question_normalize=True, 
    reduce_lr=0.0, 
    resume='', 
    resume_options=False, 
    rnn_layers=1, 
    rnn_type='lstm', 
    save_best_only=True, 
    save_for_predict=False, 
    seed=1023, 
    self_attention_opt=1, 
    task_name='QuAC', 
    train_dir='QuAC_data/', 
    tune_partial=400, 
    use_dialog_act=False, 
    use_elmo=True, 
    use_ner=True, 
    use_pos=True, 
    use_wemb=True, 
    weight_decay=0
```
result:
```shell
    Train:  F1 = 3.298,     NA = 100.000,   TH = -18.768
    Test:   F1 = 3.30,      HEQ_Q=2.74,     HEQ_D = 0.81
```

#### CoQA
parameters:
```shell
gpu1 coqa_perq0:
    CoVe_opt=1, 
    MTLSTM_path='glove/MT-LSTM.pth', 
    answer_type_num=4, 
    batch_size=1, 
    concat_rnn=False, 
    cuda=True, 
    deep_att_hidden_size_per_abstr=250, 
    deep_inter_att_do_similar=0, 
    dev_dir='CoQA/', 
    do_hierarchical_query=True, 
    do_not_save=False, 
    do_prealign=True, 
    do_ptr_update=True, 
    do_residual=False, 
    do_residual_everything=False, 
    do_residual_rnn=False, 
    do_seq_dropout=True, dropout_emb=0.4, 
    elmo_batch_size=12, 
    elmo_lambda=0.0,
    epoches=30, 
    eval_per_epoch=1, 
    explicit_dialog_ctx=1, 
    final_output_att_hidden=250, 
    fix_embeddings=False, 
    grad_clipping=10, 
    hidden_size=125, 
    learning_rate=0.1, 
    log_file='output.log', 
    log_per_updates=20, 
    max_len=15, 
    model_dir='models', 
    momentum=0, 
    my_dropout_p=0.4, 
    name='', 
    ner_dim=8, 
    ner_size=19, 
    no_dialog_flow=False, 
    no_em=False, 
    optimizer='adamax', 
    pos_dim=12, 
    pos_size=51, 
    prealign_hidden=300, 
    prealign_option=2, 
    pretrain='', 
    ptr_net_attn_type='Bilinear', 
    ptr_net_indep_attn=True, 
    question_merge='linear_self_attn', 
    question_normalize=True, 
    rationale_lambda=0.0, 
    reduce_lr=0.0, 
    resume='', 
    resume_options=False, 
    rnn_layers=1, 
    rnn_type='lstm', 
    save_best_only=True, 
    save_for_predict=False, 
    seed=1023, 
    self_attention_opt=1, 
    task_name='CoQA', 
    train_dir='CoQA/', 
    tune_partial=400, 
    use_elmo=True, 
    use_ner=True, 
    use_pos=True, 
    use_wemb=True, 
    weight_decay=0
```
result:
```shell
    Train:  F1 = 4.599,
    Test:   F1 = 4.605
```

### Environment

Tensorflow1.12 + CUDA9.0 + python3.6 + Tensor2tensor

  
Download and install [Anaconda](https://www.anaconda.com/download/#linux) 
  
You can find more install details about this on my [blog](https://blog.csdn.net/Davidddl/article/details/81873606).
  
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

<!--

##### Download extracted bert features of [PERQA](https://drive.google.com/open?id=1_KP3YOeCrpwuV8Qecq2RXyBvPkwso-7-)
|[PERQA_p1](https://drive.google.com/open?id=1yP0mbo2DI7X-CujPpAJ_TWnxM6niOqBY)|[PERQA_p2](https://drive.google.com/open?id=1vWHyzxIZm5lYtg4iThpTsqoJ4498NhxY)|
|---|---|

-->

#### GloVe
You could find our modified glove for Chinese. [glove-tools](https://github.com/Daviddddl/glove-tools)

If you want to extract your own proposal data, you could find more details from [glove-tools](https://github.com/Daviddddl/glove-tools)

Due to lack of support for Chinese in [basic projects](https://github.com/maciejkula/glove-python), we adopted [LTP](https://github.com/HIT-SCIR/ltp), [LTP_py](https://github.com/HIT-SCIR/pyltp)

LTP can be download from [Google_Drive](https://drive.google.com/open?id=1_eBscwUpr6eZPU5749H_lPikcR7FfDrB)


#### CoVE

CoVE project: [cove](https://github.com/salesforce/cove)


#### ELMo
Bidirectional LSTM-CRF and ELMo project: [anago](https://github.com/Hironsan/anago)


#### Whole project files structure
```
.
├── baseline
│   ├── DrQA
│   │   ├── download.sh
│   │   ├── drqa
│   │   │   ├── __init__.py
│   │   │   ├── pipeline
│   │   │   │   ├── drqa.py
│   │   │   │   └── __init__.py
│   │   │   ├── reader
│   │   │   │   ├── config.py
│   │   │   │   ├── data.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── layers.py
│   │   │   │   ├── model.py
│   │   │   │   ├── predictor.py
│   │   │   │   ├── rnn_reader.py
│   │   │   │   ├── utils.py
│   │   │   │   └── vector.py
│   │   │   ├── retriever
│   │   │   │   ├── doc_db.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── tfidf_doc_ranker.py
│   │   │   │   └── utils.py
│   │   │   └── tokenizers
│   │   │       ├── corenlp_tokenizer.py
│   │   │       ├── __init__.py
│   │   │       ├── regexp_tokenizer.py
│   │   │       ├── simple_tokenizer.py
│   │   │       ├── spacy_tokenizer.py
│   │   │       └── tokenizer.py
│   │   ├── img
│   │   │   └── drqa.png
│   │   ├── install_corenlp.sh
│   │   ├── LICENSE
│   │   ├── PATENTS
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── scripts
│   │   │   ├── convert
│   │   │   │   ├── squad.py
│   │   │   │   └── webquestions.py
│   │   │   ├── distant
│   │   │   │   ├── check_data.py
│   │   │   │   ├── generate.py
│   │   │   │   └── README.md
│   │   │   ├── pipeline
│   │   │   │   ├── eval.py
│   │   │   │   ├── interactive.py
│   │   │   │   └── predict.py
│   │   │   ├── reader
│   │   │   │   ├── interactive.py
│   │   │   │   ├── predict.py
│   │   │   │   ├── preprocess.py
│   │   │   │   ├── README.md
│   │   │   │   └── train.py
│   │   │   └── retriever
│   │   │       ├── build_db.py
│   │   │       ├── build_tfidf.py
│   │   │       ├── eval.py
│   │   │       ├── interactive.py
│   │   │       ├── prep_wikipedia.py
│   │   │       └── README.md
│   │   └── setup.py
│   └── FlowQA
│       ├── CoQA_eval.py
│       ├── download.sh
│       ├── general_utils.py
│       ├── predict_CoQA.py
│       ├── predict_QuAC.py
│       ├── preprocess_CoQA.py
│       ├── preprocess_QuAC.py
│       ├── QA_model
│       │   ├── detail_model.py
│       │   ├── layers.py
│       │   ├── model_CoQA.py
│       │   ├── model_QuAC.py
│       │   └── utils.py
│       ├── README.md
│       ├── requirements.txt
│       ├── train_CoQA.py
│       └── train_QuAC.py
├── bert
│   ├── chinese_L-12_H-768_A-12
│   │   ├── bert_config.json
│   │   ├── bert_model.ckpt.data-00000-of-00001
│   │   ├── bert_model.ckpt.index
│   │   ├── bert_model.ckpt.meta
│   │   └── vocab.txt
│   ├── extract_features.py
│   ├── __init__.py
│   ├── modeling.py
│   ├── __pycache__
│   │   ├── extract_features.cpython-35.pyc
│   │   ├── extract_features.cpython-36.pyc
│   │   ├── __init__.cpython-35.pyc
│   │   ├── __init__.cpython-36.pyc
│   │   ├── modeling.cpython-35.pyc
│   │   ├── modeling.cpython-36.pyc
│   │   └── tokenization.cpython-36.pyc
│   ├── run_classifier.py
│   ├── tmp_in.txt
│   ├── tmp_out.txt
│   ├── tokenization.py
│   └── tokenization_test.py
├── data
│   ├── chzhu.pkl
│   ├── dldi.pkl
│   ├── jwhu.pkl
│   ├── lmzhang.pkl
│   ├── xfduan.pkl
│   ├── xichen.pkl
│   ├── zh_session_ano.json
│   ├── zqzhu.pkl
│   └── zyzhao.pkl
├── __init__.py
├── README.md
├── requirements.txt
├── tree.txt
└── utils
    ├── __init__.py
    ├── PERQAInstance.py
    ├── prepare.py
    ├── __pycache__
    │   ├── PERQAInstance.cpython-35.pyc
    │   └── PERQAInstance.cpython-36.pyc
    ├── tmp_in.txt
    └── tmp_out.txt
```

