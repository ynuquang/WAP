#!/usr/bin/env python

import os
import sys
import pickle as pkl
import numpy
from scipy.misc import imread, imresize, imsave

image_paths = ["./off_image_train/", "./off_image_test/"]
outFiles = ["offline-train.pkl", "offline-test.pkl"]
scpFilePaths = ["train_caption.txt", "test_caption.txt"]

cases_to_avoid = ['31_em_196', '501_em_15', 'RIT_2014_210', '504_em_38', '514_em_348', 'RIT_2014_185'] # too big for our GPU welp
# min(w*h) of cases to avoid
max_pix = 120925

for image_path, outFile, scpFilePath in zip(image_paths, outFiles, scpFilePaths):
    oupFp_feature = open(outFile, "wb")
    features = {}
    channels = 1
    sentNum = 0
    scpFile = open(scpFilePath)
    while 1:
        line = scpFile.readline().strip()  # remove the '\r\n'
        if not line:
            break
        else:
            key = line.split("\t")[0]
            image_file = image_path + key + "_" + str(0) + ".bmp"
            im = imread(image_file)
            mat = numpy.zeros([channels, im.shape[0], im.shape[1]], dtype="uint8")

            if (im.shape[0] * im.shape[1] >= max_pix): # too big for our GPU welp
                continue

            for channel in range(channels):
                image_file = image_path + key + "_" + str(channel) + ".bmp"
                im = imread(image_file)
                mat[channel, :, :] = im
            sentNum = sentNum + 1
            features[key] = mat
            if round(sentNum / 500) == sentNum * 1.0 / 500:
                print("process sentences ", sentNum)

    print("load images done. sentence number ", sentNum)

    pkl.dump(features, oupFp_feature)
    print("save file done")
    oupFp_feature.close()
