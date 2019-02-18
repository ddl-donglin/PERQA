import json


def tranformer_2_quac():
    raw_data_f = open('data/zh_session_ano.json', 'r')
    raw_data_json = json.load(raw_data_f)

    for each_split in ['train', 'dev', 'test']:
        print("Generating " + each_split)
        ids_list = []
        with open('data/' + each_split + '_list.txt', 'r') as ids_f:
            ids_list.extend(ids_f.read().splitlines())

        with open('data/quac_format/perqa_' + each_split + '.json', 'w+') as perqa_raw_qa_json_f:
            qa_json_data_list = []

            for each_name in raw_data_json.keys():
                print(each_name)
                for each_ins in raw_data_json[each_name]:
                    each_data_json_id = each_name + '_' + str(each_ins['raw_id'])
                    # print(each_data_json_id)
                    if each_data_json_id in ids_list:
                        qas_list = []
                        for each_qa in each_ins['qas']:
                            span_start = '\n'.join(each_ins['session']).find(each_qa['a'])

                            if span_start == -1:
                                span_start = 0
                                sess_un = each_qa['un']
                                for each_sess in each_ins['session']:
                                    sess_un -= 1
                                    if sess_un >= 0:
                                        span_start += len(each_sess)

                            span_text = each_qa['a']
                            orig_answer_json = {
                                'answer_start': span_start,
                                'text': span_text
                            }

                            qa_json = {
                                'followup': 'n',
                                'yesno': 'x',
                                'question': each_qa['q'],
                                'answers': [orig_answer_json],
                                'id': each_data_json_id + '_' + str(each_qa['un']),
                                'orig_answer': orig_answer_json
                            }
                            qas_list.append(qa_json)

                        paragraphs_json = {
                            'context': '\n'.join(each_ins['session']),
                            'qas': qas_list,
                            'id': each_data_json_id
                        }
                        paragraphs_list = [paragraphs_json]

                        each_data_json = {
                            'title': each_data_json_id,
                            'paragraphs': paragraphs_list
                        }
                        qa_json_data_list.append(each_data_json)

            qa_json = {"data": qa_json_data_list}
            json.dump(qa_json, perqa_raw_qa_json_f, ensure_ascii=False)
    raw_data_f.close()


if __name__ == '__main__':
    with open('data/quac_format/perqa_train.json', 'r') as f:
        json_data = json.load(f)

    # print(json_data.keys())     # dict_keys(['data'])
    # level_1 = json.dumps(json_data['data'][0])
    # print(json.loads(level_1).keys())       # dict_keys(['title', 'paragraphs'])
    print(json.dumps(json_data['data'][0], ensure_ascii=False))

    # tranformer_2_quac()


