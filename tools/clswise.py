
import argparse
from statistics import mean
import os


def scores_from_log(logfile):
    # print(os.listdir('.'))
    with open(logfile) as f:
        lines = f.readlines()
    scoreline = lines[-1]
    cls_wises = [list(map(float,clswise.split('[')[1].split(']')[0].split(','))) for clswise in scoreline.split('classwise')[1:]]

    return cls_wises


def main():
    SNIFFYART_CLASSES = [
        'cooking', 'dancing', 'drinking', 'eating', 'holding the nose', 'painting', 'peeing',
        'playing music', 'praying', 'reading', 'sleeping', 'smoking', 'sniffing', 'textile work',
        'writing', 'none'
    ]

    v1_labels = ['cooking', 'drinking', 'smoking', 'holding the nose', 'sniffing', 'none']

    v1_idxs = [SNIFFYART_CLASSES.index(c) for c in v1_labels]

    parser = argparse.ArgumentParser()
    parser.add_argument('logfile')
    args = parser.parse_args()

    metrics = scores_from_log(args.logfile)

    for scores in metrics:
        print(mean([scores[i] for i in v1_idxs]))

if __name__ == '__main__':
    main()