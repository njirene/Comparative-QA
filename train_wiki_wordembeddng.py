#!/home/liuzheng/miniconda3/bin/python
# -*- coding: utf-8 -*-
import os, sys
import multiprocessing
import gensim  
import time

def word2vec_train(input_file, output_file):
    sentences = gensim.models.word2vec.LineSentence(input_file)
    model = gensim.models.Word2Vec(sentences, size=300, min_count=10, iter=20, sg=0, workers=multiprocessing.cpu_count())
    # 2016-03-20 11:45:39 time format
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) 
    model.save(output_file)
    model.wv.save_word2vec_format(output_file + '.model.bin', binary=True)     
    model.wv.save_word2vec_format(output_file + '.model.txt', binary=False)
    #model.save_word2vec_format(output_file + '.vector', binary=True)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python script.py infile outfile")
        sys.exit()
    print("start training--------------")
    # 2016-03-20 11:45:39 time format
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) 
    time_start = time.time()
    input_file, output_file = sys.argv[1], sys.argv[2]
    word2vec_train(input_file, output_file)
    print("time cost: %fm" %((time.time()-time_start)/60))
