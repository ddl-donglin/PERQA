import json


def transform_2_coqa():
    raw_data_f = open('data/zh_session_ano.json', 'r')
    raw_data_json = json.load(raw_data_f)
    for each_split in ['train', 'dev', 'test']:
        print("Generating " + each_split)
        ids_list = []
        with open('data/' + each_split + '_list.txt', 'r') as ids_f:
            ids_list.extend(ids_f.read().splitlines())

        # print(each_split + str(len(ids_list)))
        with open('data/perqa_' + each_split + '.json', 'w+') as perqa_raw_qa_json_f:
            qa_json = {"version": "1.0"}
            qa_json_data_list = []

            for each_name in raw_data_json.keys():
                print(each_name)
                for each_ins in raw_data_json[each_name]:
                    each_data_json_id = each_name + '_' + str(each_ins['raw_id'])
                    # print(each_data_json_id)
                    if each_data_json_id in ids_list:
                        questions_list = []
                        answers_list = []

                        for each_qa in each_ins['qas']:
                            each_q_json = {'input_text': each_qa['q'], 'turn_id': each_qa['un']}
                            span_start = '\n'.join(each_ins['session']).find(each_qa['a'])
                            if span_start == -1:
                                span_start = 0
                                sess_un = each_qa['un']
                                for each_sess in each_ins['session']:
                                    sess_un -= 1
                                    if sess_un >= 0:
                                        span_start += len(each_sess)
                                span_end = span_start + each_ins['session'][each_qa['un'] - 1]
                                span_text = each_qa['a']
                            else:
                                span_end = span_start + len(each_qa['a'])
                                span_text = each_qa['a']

                            each_a_json = {'span_start': span_start,
                                           'span_end': span_end,
                                           'span_text': span_text,
                                           'input_text': each_ins['session'][each_qa['un']-1],
                                           'turn_id': each_qa['un']
                                            }

                            questions_list.append(each_q_json)
                            answers_list.append(each_a_json)

                        each_data_json = {
                            "source": "douban",
                            'id': each_data_json_id,
                            'filename': "null",
                            "story": '\n'.join(each_ins['session']),
                            "questions": questions_list,
                            "answers": answers_list,
                            "name": each_data_json_id
                        }

                        qa_json_data_list.append(each_data_json)

            qa_json['data'] = qa_json_data_list
            json.dump(qa_json, perqa_raw_qa_json_f, ensure_ascii=False)
    raw_data_f.close()


if __name__ == '__main__':
    transform_2_coqa()
