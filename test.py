#-*- coding=utf-8 -*-
import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from extract_cnn_vgg16_keras import VGGNet

query = 'test.jpg'
index = 'models/vgg_featureCNN.h5'
result = 'database'
# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(index, 'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")

# read and show query image
queryImg = mpimg.imread(query)
plt.title("Query Image")
plt.imshow(queryImg)
plt.show()

# init VGGNet16 model
model = VGGNet()

# extract query image's feature, compute simlarity score and sort
queryVec = model.vgg_extract_feat(query)  # 修改此处改变提取特征的网络
# print(queryVec.shape)
# print(feats.shape)
print('--------------------------')
# print(queryVec)
# print(feats.T)
print('--------------------------')
scores = np.dot(queryVec, feats.T)
# scores = np.dot(queryVec, feats.T)/(np.linalg.norm(queryVec)*np.linalg.norm(feats.T))
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]
# print (rank_ID)
print(rank_score)

# number of top retrieved images to show
maxres = 3  # 检索出三张相似度最高的图片
imlist = []
for i, index in enumerate(rank_ID[0:maxres]):
    imlist.append(imgNames[index])
    # print(type(imgNames[index]))
    print("image names: " + str(imgNames[index]) + " scores: %f" % rank_score[i])
print("top %d images in order are: " % maxres, imlist)
# show top #maxres retrieved result one by one
for i, im in enumerate(imlist):
    image = mpimg.imread(result + "/" + str(im, 'utf-8'))
    plt.title("search output %d" % (i + 1))
    plt.imshow(image)
    plt.show()
