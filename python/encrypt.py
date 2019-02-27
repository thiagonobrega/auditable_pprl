'''
Created on 9 de abr de 2017

@author: Thiago
'''

import ngram
from lib.mybloom.bloomfilter import BloomFilter
import bitarray


def encryptData(data,size,fp=0.01,n=2):
    """
        n : 2 = Bigrams
        size : Size of BF
        fp : False positive rate
    """
    bloomfilter = BloomFilter(size,fp)
    index = ngram.NGram(N=n)
    bigrams = list(index.ngrams(index.pad(str(data))))

    for bigram in bigrams:
        bloomfilter.add(str(bigram))

    return bloomfilter

# '0xC1844bbe0537cE51F95F9EC08c55D697fCcf3f17'
def bitarray2int(ba):
    import struct
    return int(ba.to01(), 2)

def int2bitarray(num):
    return bitarray.bitarray(num)


# bitarray2int(bf.filter)

import csv


base_dir = '/home/thiagonobrega/workspace/bc-playground/Datasets/bikes/'
lista = []

with open(base_dir+'candset.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            dbf1 = row[3] + row[4] + row[7]
            dbf2 = row[8] + row[9] + row[12]

            linha = [row[1],encryptData(dbf1,128,n=3),row[2],encryptData(dbf2,128,n=3)]
            # print(f'\tlt_id={row[1]} name={row[3]}, city={row[4]}, color={row[7]}')
            # print(f'\tlt_id={row[2]} name={row[8]}, city={row[9]}, color={row[12]}')
            lista.append(linha)
            line_count += 1
    print(f'Processed {line_count} lines.')
