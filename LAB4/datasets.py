from conll_dictorizer import CoNLLDictorizer, Token
import os

def load_conll2009_pos():
    train_file = 'datasets\train.txt'
    dev_file = 'datasets\valid.txt'
    test_file = 'datasets\test.txt'
    test2_file = 'simple_pos_test.txt'

    column_names = ['id', 'form', 'lemma', 'plemma', 'pos', 'ppos']

    train_sentences = open(train_file).read().strip()
    dev_sentences = open(dev_file).read().strip()
    test_sentences = open(test_file).read().strip()
    test2_sentences = open(test2_file).read().strip()
    return train_sentences, dev_sentences, test_sentences, column_names

def load_conll2003_en():
    BASE_DIR = os.getcwd()
    train_file = BASE_DIR + '\\datasets\\train.txt'
    dev_file = BASE_DIR + '\\datasets\\valid.txt'
    test_file = BASE_DIR + '\\datasets\\test.txt'
    column_names = ['form', 'ppos', 'pchunk', 'ner']
    train_sentences = open(train_file).read().strip()
    dev_sentences = open(dev_file).read().strip()
    test_sentences = open(test_file).read().strip()
    return train_sentences, dev_sentences, test_sentences, column_names


if __name__ == '__main__':
    train_sentences, dev_sentences, test_sentences, column_names = load_conll2003_en()

    conll_dict = CoNLLDictorizer(column_names, col_sep=' +')
    train_dict = conll_dict.transform(train_sentences)
    print(train_dict[0])
    print(train_dict[1])