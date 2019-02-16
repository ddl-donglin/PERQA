import json
import pickle
import PERQAInstance
from bert.extract_features import my_get_features
import numpy

root_path = '/home/david/PycharmProjects/PERQA/data/'
name_list = ['xfduan', 'dldi', 'lmzhang', 'zyzhao', 'zqzhu', 'chzhu', 'jwhu', 'xichen']
"""
    xfduan:     736
    dldi:       804
    lmzhang:    802
    zyzhao:     920
    zqzhu:      804
    chzhu:      702
    jwhu:       729
    xichen:     706
    all:        6203
"""


def gen_bert_features(input_str):
    # cmd1 = 'source /root/conda activate tensorflow'
    # cmd2 = 'BERT_BASE_DIR=' + '/home/david/PycharmProjects/PERQA/encoding/bert/chinese_L-12_H-768_A-12'
    # cmd3 = 'python /home/david/PycharmProjects/PERQA/encoding/bert/extract_features.py ' \
    #        ' --input_file=' + input_path + \
    #        ' --output_file=' + output_path + \
    #        ' --vocab_file=$BERT_BASE_DIR/vocab.txt' \
    #        ' --bert_config_file=$BERT_BASE_DIR/bert_config.json' \
    #        ' --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt' \
    #        ' --layers=-1,-2,-3,-4   --max_seq_length=128   --batch_size=8'
    # subprocess.run("bash -c " + cmd1 + ' , shell=True')
    # os.system(cmd1 + ' && ' + cmd2 + ' && ' + cmd3)
    return my_get_features(input_str)


def gen_glove_features(input_path, output_path):
    features = []
    return features


def gen_cove_features(input_path, output_path):
    features = []
    return features


def gen_elmo_features(input_path, output_path):
    features = []
    return features


def gen_all_instances(json_path=root_path + 'zh_session_ano.json', feature_type='bert'):
    """
    feature_type: bert, GloVe, CoVE, ELMo
    :param json_path:
    :param feature_type:
    :return:
    """
    f = open(json_path, 'r')
    json_data = json.loads(f.read())
    for each_name in json_data.keys():
        all_num = len(json_data[each_name])
        idx = 0
        # perqa_ins_list = []
        each_name_perqa_ins_f = open(root_path + each_name + '.pkl', 'wb+')

        for each_case in json_data[each_name]:
            session = each_case['session']
            raw_id = each_case['raw_id']
            qas = each_case['qas']
            idx += 1
            print(each_name, all_num, idx, raw_id)

            # generate bert sess features
            sess_features = []
            for each_line in session:
                if feature_type == 'bert':
                    sess_features.append(gen_bert_features(each_line))

            qas_f = []
            for each_qa in qas:
                un = each_qa['un']
                q_str = each_qa['q']
                a_str = each_qa['a']

                # generate bert qa features
                q_features = gen_bert_features(q_str)
                a_features = gen_bert_features(a_str)

                qas_f.append((un, q_features, a_features))

            # generate PERQAInstance
            perqa_ins = PERQAInstance.PERQAInstance(
                name=each_name,
                session=session,
                raw_id=raw_id,
                session_f=sess_features,
                qas_f=qas_f
            )

            # perqa_ins_list.append(perqa_ins)
            pickle.dump(perqa_ins, each_name_perqa_ins_f)
        each_name_perqa_ins_f.close()

        # with open(each_name + '.pkl', 'wb+') as perqa_f:
        #     pickle.dump(perqa_ins_list, perqa_f)


def load_instances(ins_path, get_ins_num=-1):
    """
    root_path + name_list + '.pkl'
    root_path + train.pkl / dev.pkl / test.pkl

    :param ins_path:
    :param get_ins_num: -1 means get all instances list
    :return:
    """
    instances_list = []
    with open(ins_path, 'rb') as load_f:
        print("Loading instances list..." + ins_path)
        if get_ins_num == -1:
            # get all instances
            while True:
                try:
                    instances_list.append(pickle.load(load_f))
                except EOFError:
                    print('The pkl file ' + ins_path + ' ends.')
                    break
        else:
            while get_ins_num > 0:
                try:
                    instances_list.append(pickle.load(load_f))
                    get_ins_num -= 1

                except EOFError:
                    print('The pkl file ' + ins_path + ' ends.')
                    break
    return instances_list


def split_dataset(dataset_path=root_path):
    """
    generate train, dev and test instances lists
    :param dataset_path:
    :return:
    """
    # train_list = []
    # dev_list = []
    # test_list = []
    train_num = 0
    dev_num = 0
    test_num = 0

    f = open(dataset_path + 'zh_session_ano.json', 'r')
    json_data = json.loads(f.read())
    # all_ins_num = 0
    # for each_name in json_data:
    #     all_ins_num += len(json_data[each_name])
    # print(all_ins_num)    # all: 6203

    all_ins_list = []
    for each_name in json_data:
        for each_ins in json_data[each_name]:
            all_ins_list.append((each_name, each_ins['raw_id']))

    # print(all_ins_list[3])      # ('xfduan', 6)
    all_ins_num = len(all_ins_list)
    # print(all_ins_num)    # 6203

    numpy.random.shuffle(all_ins_list)
    train, dev, test = all_ins_list[: int(all_ins_num * 0.8)], \
                       all_ins_list[int(all_ins_num * 0.8): int(all_ins_num * 0.9)], \
                       all_ins_list[int(all_ins_num * 0.9):]

    # print(len(train), len(dev), len(test))  # 4962 620 621

    train_f = open(root_path + 'train.pkl', 'wb+')
    dev_f = open(root_path + 'dev.pkl', 'wb+')
    test_f = open(root_path + 'test.pkl', 'wb+')

    for each_name in name_list:
        for each_ins in load_instances(root_path + each_name + '.pkl'):
            if (each_ins.name, each_ins.raw_id) in train:
                # train_list.append(each_ins)
                train_num += 1
                pickle.dump(each_ins, train_f)
            elif (each_ins.name, each_ins.raw_id) in dev:
                # dev_list.append(each_ins)
                dev_num += 1
                pickle.dump(each_ins, train_f)
            elif (each_ins.name, each_ins.raw_id) in test:
                # test_list.append(each_ins)
                test_num += 1
                pickle.dump(each_ins, train_f)
            else:
                print(str((each_ins.name, each_ins.raw_id)) + ' is wrong !?????')

    train_f.close()
    test_f.close()
    dev_f.close()
    # print(train_num, dev_num, test_num)     # 4962 620 621
    # print(len(train_list))
    # print(len(dev_list))
    # print(len(test_list))


if __name__ == '__main__':
    # visualization

    # show_ins_num = 2
    # for each_ins in load_instances(root_path + 'xfduan.pkl', show_ins_num):
    #     if show_ins_num > 0:
    #         print(each_ins.name)
    #         print(each_ins.session)
    #         print(each_ins.raw_id)
    #         print("session_f: ")
    #         for each_f in each_ins.session_f:
    #             print(len(each_f))
    #         print("qas_f: ")
    #         for each_f in each_ins.qas_f:
    #             print(each_f[0], len(each_f))
    #
    #         show_ins_num -= 1
    #     else:
    #         break

    """
    xfduan
    ['总觉得自己去菜场买菜很有画面感啊！', 
     '啊…上次心血来潮去卖暖贴 卖的好惨好惨啊 lz咋卖？', 
     '啊？我咋卖？我不卖东西诶 我是去买东西哦',
     '那 你买东西的时候有木有挎着框', 
     '哈哈哈哈 没有诶', 
     '那就太没画面感了…']
    1
    session_f: 
    58368
    76800
    64512
    49152
    27648
    33792
    qas_f: 
    5 3
    6 3
    xfduan
    ['mlgb刚才脚板抽筋怎么算。', '缺钙？', '喝蓝瓶娃哈哈。', '买几片钙片吃吃', '有用吗！', '试试吧 反正不贵 吃了没坏处']
    2
    session_f: 
    43008
    15360
    27648
    27648
    18432
    43008
    qas_f: 
    3 3
    5 3
    6 3
    """

    # gen_all_instances()

    split_dataset()
