import os
import subprocess
import json
import pickle
import PERQAInstance

json_path = '/home/david/PycharmProjects/PERQA/data/zh_session_ano.json'


def gen_bert_features(input_path, output_path):
    cmd1 = 'source /root/conda activate tensorflow'
    cmd2 = 'BERT_BASE_DIR=' + '/home/david/PycharmProjects/PERQA/encoding/bert/chinese_L-12_H-768_A-12'
    cmd3 = 'python /home/david/PycharmProjects/PERQA/encoding/bert/extract_features.py ' \
           ' --input_file=' + input_path + \
           ' --output_file=' + output_path + \
           ' --vocab_file=$BERT_BASE_DIR/vocab.txt' \
           ' --bert_config_file=$BERT_BASE_DIR/bert_config.json' \
           ' --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt' \
           ' --layers=-1,-2,-3,-4   --max_seq_length=128   --batch_size=8'
    subprocess.run("bash -c " + cmd1 + ' , shell=True')
    # os.system(cmd1 + ' && ' + cmd2 + ' && ' + cmd3)


def gen_glove_features(input_path, output_path):
    features = []
    return features


def gen_cove_features(input_path, output_path):
    features = []
    return features


def gen_elmo_features(input_path, output_path):
    features = []
    return features


def gen_all_instances(json_path, feature_type):
    """
    feature_type: GloVe, CoVE, ELMo
    :param json_path:
    :param feature_type:
    :return:
    """
    f = open(json_path, 'r')
    json_data = json.loads(f.read())
    for each_name in json_data.keys():
        all_num = len(json_data[each_name])
        idx = 0
        perqa_ins_list = []

        for each_case in json_data[each_name]:
            session = each_case['session']
            raw_id = each_case['raw_id']
            qas = each_case['qas']
            idx += 1
            print(each_name, all_num, idx, raw_id)

            with open('tmp_sess.txt', 'w+') as sess_f:
                for each_line in session:
                    sess_f.write(each_line)

            qas_f = []
            for each_qa in qas:
                un = each_qa['un']
                with open('tmp_q.txt', 'w+') as q_f:
                    q_f.write(each_qa['q'])
                with open('tmp_a.txt', 'w+') as a_f:
                    a_f.write(each_qa['a'])

                # generate bert qa features (list)

            # generate bert sess features

            # generate PERQAInstance
            perqa_ins = PERQAInstance.PERQAInstance(
                name=each_name,
                session=session,
                raw_id=raw_id,
                session_f=sess_f,
                qas_f=qas_f
            )

            perqa_ins_list.append(perqa_ins)

        with open(each_name+'.pkl', 'wb+') as perqa_f:
            pickle.dump(perqa_ins_list, perqa_f)


if __name__ == '__main__':
    gen_bert_features('test_in.txt', 'test_out.json')
