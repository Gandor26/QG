import argparse
import prep

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', nargs='?', default='/home/x_jin/workspace/nlg/data', type=str)
    parser.add_argument('--word2vec', '-t', nargs='?', default='/mnt/data/GloVe/GloVe_840B.bin', type=str)
    args = parser.parse_args()

    data_file = 'squad-v1.1.json'

    config = dict(
        data_folder=args.folder,
        word2vec_path=args.word2vec,
        )
    dp = prep.DataPrepare(**config)
    dp.process(data_file)

