"""Script for converting v3 .csv to v5 .pickle.gz"""
import argparse
import gzip
import os
import pickle

from tesserae.tokenizers import tokenizer_map


def _main():
    args = _parse_args()
    tokenizer = tokenizer_map[args.language](None)
    data = _read_csv(args.csv_file, tokenizer.normalize)
    _write_picklegz(data, args.csv_file)


def _parse_args():
    p = argparse.ArgumentParser(
        prog='tesserae.data.create',
        description='Create a .pickle.gz file from a .csv')
    p.add_argument('language', choices=[lang for lang in tokenizer_map])
    p.add_argument('csv_file')
    return p.parse_args()


def _read_csv(filename, normalizer):
    result = {}
    with open(filename, 'r', encoding='utf-8') as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                items, _ = normalizer(line)
                if len(items) >= 2:
                    result[items[0]] = items[1:]
    return result


def _write_picklegz(data, filename):
    data_dir = os.path.dirname(__file__)
    filebasename = os.path.basename(filename)[:-4]
    new_name = os.path.join(data_dir, filebasename + '.pickle.gz')
    with gzip.GzipFile(new_name, 'w') as ofh:
        pickle.dump(data, ofh)


if __name__ == '__main__':
    _main()
