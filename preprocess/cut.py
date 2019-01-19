import re
import json
import spacy
import numpy as np
import os.path as path
from tqdm import tqdm
from collections import Counter, Iterable

def save_file(data, data_file):
    print('Saving %s' % data_file)
    with open(data_file, 'w') as fp:
        json.dump(data, fp)
    print('Saved %s' % data_file)

def load_file(data_file):
    print('Loading %s' % data_file)
    with open(data_file, 'r') as fp:
        data = json.load(fp)
    print('Loaded %s' % data_file)
    return data

def load_data(data_file):
    '''
        data = [{context, [{[{answer_text, answer_pos}], question}]}]
    '''
    print('Loading raw data from %s' % data_file)
    data = []
    raw = load_file(data_file)['data']
    for article in raw:
        data.extend(article['paragraphs'])
    print('Loaded dataset')
    return data

if __name__ == '__main__':
    trans = str.maketrans({
    '(': ' (',
    '[': ' [',
    '{': ' {',
    ')': ') ',
    ']': '] ',
    '}': '} '
    })
    tkz = spacy.load('en')
    data = load_data('../data/squad-v1.1.json')
    docs = []
    questions = []
    answers = []
    for paragraph in data:
        doc = paragraph['context']
        for qa in paragraph['qas']:
            questions.append(qa['question'])
            answer_dict = qa['answer']
            answers.append(answer_dict['text'].translate(trans))
            docs.append(' @ans_b@ '.join([doc[:answer_dict['answer_start']], doc[answer_dict['answer_start']:]]).translate(trans))
    assert len(docs)==len(answers)
    assert len(docs)==len(questions)
    num = len(docs)
    doc_pipe = tkz.pipe(docs, n_threads=4)
    question_pipe = tkz.pipe(questions, n_threads=4)
    answer_pipe = tkz.pipe(answers, n_threads=4)
    docs = []
    questions = []
    answers = []
    for doc, question, answer in tqdm(zip(doc_pipe, question_pipe, answer_pipe), total=num):
        answer_tokens = [w.lower_ for w in answer if not w.is_space]
        doc_tokens = [w.lower_ for w in doc if not w.is_space]
        question_tokens = [w.lower_ for w in question if not (w.is_space or w.is_punct)]
        answer_start = doc_tokens.index('@ans_b@')
        answer_end = answer_start + len(answer_tokens)
        doc_tokens.insert(answer_end+1, '@ans_e@')
        docs.append(doc_tokens)
        questions.append(question_tokens)
        answers.append([w.lower_ for w in answer])
    f = open('../data/pairs.txt', 'w')
    for doc, question, answer in zip(docs, questions, answers):
        doc_str = ' '.join(doc)
        question_str = ' '.join(question)
        answer_str = ' '.join(answer)
        pair_str = '\n'.join([doc_str, 'QUESTION:\t'+question_str, 'ANSWER:\t'+answer_str, '-'*100])
        f.write(pair_str+'\n')

