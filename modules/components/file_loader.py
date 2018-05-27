import pandas as pd

basepath = './data/'


def csv_loader(filename):
    return pd.read_csv(basepath + filename)


def load_file():
    while True:
        try:
            filename = raw_input('Filename (.csv): ')
            return csv_loader(filename)
        except IOError as e:
            print 'I/O error({0}) - Cannot find file: {1}'.format(e.errno, filename)
