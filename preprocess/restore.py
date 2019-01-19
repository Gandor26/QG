import os.path
import json
import csv

def pair_generator(pairs, words, vocab):
    for pair in pairs:
        q1 = ' '.join([vocab[str(word)] for word in words[pair['question1']]])
        q2 = ' '.join([vocab[str(word)] for word in words[pair['question2']]])
        label = pair['is_duplicate']
        yield q1, q2, label

def write_file(data, filename):
    with open(filename, 'w', newline='') as fp:
        for q, a, g in data:
            fp.write('QUESTION:\t'+q+'\n')
            fp.write('ANSWER:\t'+a+'\n')
            fp.write('GROUND_TRUTH:\t'+g+'\n')
            fp.write('-'*100+'\n')

def read_file(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

if __name__ == '__main__':
    ROOT    = '/home/x_jin/workspace/nlg/data'
    DATASET = 'data.json'
    SOURCE_VOCAB = 'source_vocab.json'
    SOURCE_INDEX = 'source_index.json'
    TARGET_VOCAB = 'target_vocab.json'
    TARGET_INDEX = 'target_index.json'
    QA = 'id2qa.json'
    OUTPUT  = 'restore.txt'
    source_vocab = read_file(os.path.join(ROOT, SOURCE_VOCAB))
    target_vocab = read_file(os.path.join(ROOT, TARGET_VOCAB))
    source = read_file(os.path.join(ROOT, SOURCE_INDEX))
    target = read_file(os.path.join(ROOT, TARGET_INDEX))
    gt_lookup = read_file(os.path.join(ROOT, QA))
    data_table = read_file(os.path.join(ROOT, DATASET))
    data = []
    for d_id, q_id, answer_start, answer_length in data_table:
        gt = gt_lookup[str(q_id)][0]['text']
        answer = ' '.join([source_vocab[str(w_id)] for w_id in source[d_id][answer_start:(answer_start+answer_length)]])
        question = ' '.join([target_vocab[str(w_id)] if w_id<len(target_vocab) else source_vocab[str(source[d_id][w_id-len(target_vocab)])] for w_id in target[q_id]])
        data.append((question, answer, gt))
    write_file(data, os.path.join(ROOT, OUTPUT))


