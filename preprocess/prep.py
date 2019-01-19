import os
import json
import spacy
import enchant
import pdb
import numpy as np
import os.path as path
from tqdm import tqdm
from multiprocessing import pool
from collections import Counter, Iterable

def check_replaceable(word, max_dist=2):
    if word.isalpha() and not DataPrepare.spellchecker.check(word):
        s = DataPrepare.spellchecker.suggest(word)
        if s and edit_dist(word, s[0]) <= max_dist:
            emb = sum([w.vector for w in DataPrepare.tokenizer(s[0]) if w.has_vector])
            if isinstance(emb, np.ndarray):
                return word, emb
    return None, None

def edit_dist(str1, str2, unit_cost=1, transpositions=False):
    len1, len2 = len(str1)+1, len(str2)+1
    lev = [[0]*len2 for _ in range(len1)]
    for i in range(len1):
        lev[i][0] = i
    for j in range(len2):
        lev[0][j] = j
    for i in range(1, len1):
        for j in range(1, len2):
            a = lev[i-1][j]+1
            b = lev[i][j-1]+1
            c = lev[i-1][j-1] + (unit_cost if str1[i-1]!=str2[j-1] else 0)
            d = c+1
            if transpositions and i>1 and j>1:
                if str1[i-2]==str2[j-1] and str1[i-1]==str2[j-2]:
                    d = lev[i-2][j-2]+1
            lev[i][j] = min(a,b,c,d)
    return lev[-1][-1]


class DataPrepare(object):
    tokenizer = spacy.load('en')
    spellchecker = enchant.Dict('en_US')
    trans = str.maketrans({
        '(': ' (',
        '[': ' [',
        '{': ' {',
        ')': ') ',
        ']': '] ',
        '}': '} '
        })
    def __init__(self, **config):
        self.emb_dim = 300
        for key in config:
            if key == 'word2vec_path':
                print('Loading pretrained word vectors...')
                self.emb_dim = DataPrepare.tokenizer.vocab.load_vectors_from_bin_loc(config[key])
                print('Finish loading word vectors.')
            else:
                setattr(self, key, config[key])

    def _save_file(self, data, data_file):
        print('Saving %s' % data_file)
        with open(path.join(self.data_folder, data_file), 'w') as fp:
            json.dump(data, fp)
        print('Saved %s' % data_file)

    def _load_file(self, data_file):
        print('Loading %s' % data_file)
        with open(path.join(self.data_folder, data_file), 'r') as fp:
            data = json.load(fp)
        print('Loaded %s' % data_file)
        return data

    def _load_data(self, data_file):
        '''
            data = [{context, [{[{answer_text, answer_pos}], question}]}]
        '''
        print('Loading raw data from %s' % data_file)
        data = []
        raw = self._load_file(data_file)['data']
        for article in raw:
            data.extend(article['paragraphs'])
        print('Loaded dataset')
        return data

    def process(self, data_file):
        print('Processing data...')
        data = self._load_data(path.join(self.data_folder, data_file))

        lookup = [0]
        docs = []
        answers = []
        pos_no = []
        questions = []

        def seg_str(s, *cuts):
            if len(cuts)==1:
                return [s[:cuts[0]], s[cuts[0]:]]
            else:
                return [s[:cuts[0]]] + seg_str(s[cuts[0]:], *[pos-cuts[0] for pos in cuts[1:]])

        for para in data:
            doc = para['context']
            cuts = sorted(set(qa['answer']['answer_start'] for qa in para['qas']))
            for qa in para['qas']:
                questions.append(qa['question'])
                answers.append(qa['answer']['text'].translate(DataPrepare.trans))
                pos_no.append(cuts.index(qa['answer']['answer_start']))
            doc_segs = seg_str(doc, *cuts)
            tag_doc = ' @ans_b@ '.join(doc_segs)
            docs.append(tag_doc.translate(DataPrepare.trans))
            lookup.append(len(questions))

        source, target, data = [], [], []
        pipeline = DataPrepare.tokenizer.pipe(docs, n_threads=4, batch_size=500)
        for doc_id, doc in tqdm(enumerate(pipeline), desc='Tokenizing documents', total=len(docs)):
            tokens = []
            answers_pos = []
            idx = 0
            for token in doc:
                if token.is_space:
                    continue
                elif token.text == '@ans_b@':
                    answers_pos.append(idx)
                else:
                    tokens.append(token.lower_)
                    idx += 1
            for answer_id in range(lookup[doc_id], lookup[doc_id+1]):
                answer = answers[answer_id]
                answer_pos = answers_pos[pos_no[answer_id]]
                answer_len = len(DataPrepare.tokenizer(answer))
                data.append((doc_id, answer_id, answer_pos, answer_len))
            source.append(tokens)
        pipeline = DataPrepare.tokenizer.pipe(questions, n_threads=4, batch_size=500)
        for doc in tqdm(pipeline, desc='Tokenizing questions', total=len(questions)):
            words = [word.lower_ for word in doc if not word.is_punct and not word.is_space]
            target.append(words)
        self._save_file(data, 'data.json')
        self._save_file(source, 'source.tmp')
        self._save_file(target, 'target.tmp')
        #data = self._load_file('data.json')
        #source = self._load_file('source.tmp')
        #target = self._load_file('target.tmp')

        #   Building source vabulary    #
        print('Building source vocabulary...')
        vocab = set()
        for line in source:
            vocab.update(line)
        source_vocab = dict([('<PAD>', 0)])
        source_pretrained_embs = [np.zeros([self.emb_dim])]
        for word in tqdm(vocab, desc='Collect valid words'):
            token = DataPrepare.tokenizer(word)[0]
            if token.has_vector and not token.is_oov:
                source_vocab[word] = len(source_vocab)
                source_pretrained_embs.append(token.vector)
        oovs = vocab - set(source_vocab.keys())
        print('In-Vocabulary words: %d, Out-of-Vocabulary words: %d' % (len(source_vocab), len(oovs)))
        print('Start processing oov words...')
        print('First check spelling...')
        pbar = tqdm(total=len(oovs), desc='Spell checking')
        p = pool.Pool()
        oov_processed = set()
        for word, emb in p.imap(check_replaceable, oovs):
            pbar.update()
            if (not(word is None)) and (not(emb is None)):
                oov_processed.add(word)
                source_vocab[word] = len(source_vocab)
                source_pretrained_embs.append(emb)
        p.close()
        p.join()
        pbar.close()
        source_pretrained_embs = np.array(source_pretrained_embs)
        assert source_pretrained_embs.shape[0] == len(source_vocab)
        np.save(path.join(self.data_folder, 'source_embed.npy'), source_pretrained_embs)
        oovs -= oov_processed
        print('After spellcheck there are %d oov words...' % len(oovs))
        for word in tqdm(oovs, desc='Collect OOVs'):
            source_vocab[word] = len(source_vocab)
        self._save_file(dict((v, k) for k, v in source_vocab.items()), 'source_vocab.json')
        #source_vocab = dict((v,k) for k, v in self._load_file('source_vocab.json').items())

        #building document corpus
        source_index = list()
        for line in tqdm(source, desc='Indexing source words'):
            source_index.append([source_vocab[word] for word in line])
        self._save_file(source_index, 'source_index.json')

        #   building output vocabulary
        print('Building target vocabulary...')
        vocab = set()
        for doc_id, q_id, _, _ in data:
            doc = source[doc_id]
            q = target[q_id]
            vocab.update([qw for qw in q if not qw in doc])
        target_vocab = dict([('<PAD>', 0)])
        target_pretrained_embs = [np.zeros(self.emb_dim)]
        for word in tqdm(vocab, desc='Collect valid words'):
            token = DataPrepare.tokenizer(word)[0]
            if token.has_vector and not token.is_oov:
                target_vocab[word] = len(target_vocab)
                target_pretrained_embs.append(token.vector)
        oovs = vocab-set(target_vocab.keys())
        print('After spellcheck there are %d oov words...' % len(oovs))
        print('Start processing oov words...')
        print('First check spelling...')
        pbar = tqdm(total=len(oovs), desc='Spell checking')
        p = pool.Pool()
        oov_processed = set()
        for word, emb in p.imap(check_replaceable, oovs):
            pbar.update()
            if (not(word is None)) and (not(emb is None)):
                oov_processed.add(word)
                target_vocab[word] = len(target_vocab)
                target_pretrained_embs.append(emb)
        p.close()
        p.join()
        pbar.close()
        target_pretrained_embs = np.array(target_pretrained_embs)
        assert target_pretrained_embs.shape[0] == len(target_vocab)
        np.save(path.join(self.data_folder, 'target_embed.npy'), target_pretrained_embs)
        oovs -= oov_processed
        print('After spellcheck there are %d oov words...' % len(oovs))
        for word in tqdm(oovs, desc='Collect OOVs'):
            target_vocab[word] = len(target_vocab)
        target_vocab['<EOS>'] = len(target_vocab)
        self._save_file(dict((v, k) for k, v in target_vocab.items()), 'target_vocab.json')

        #   building question corpus
        target_index = list()
        for doc_id, q_id, answer_start, answer_end in tqdm(data, desc='Indexing target words'):
            doc = source[doc_id]
            line = []
            for qword in target[q_id]:
                if qword in doc:
                    line.append(len(target_vocab)+min([i for i, w in enumerate(doc) if w==qword], key=lambda x:min(abs(x-answer_start), abs(x-answer_end))))
                else:
                    line.append(target_vocab[qword])
            line.append(target_vocab['<EOS>'])
            target_index.append(line)
        self._save_file(target_index, 'target_index.json')

#if __name__ == '__main__':
#    dp = DataPrepare(data_folder='../data',
#            start_id=0,
#            word2vec_path='/mnt/data/GloVe/GloVe_840B.bin')
#    dp.process('squad-v1.1.json')
