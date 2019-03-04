#coding: utf-8
from PIL import Image
import os
import time
import numpy as np
from functools import partial
import multiprocessing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sub_modules import my_hog
from sub_modules import my_lbp
from sub_modules import my_glcm
from sub_modules import my_hessian
from sub_modules import my_pca
from sub_modules import my_canny

import warnings
warnings.filterwarnings("ignore")


try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

try:
    import cPickle as pickle
except ImportError:
    import pickle


def proxy(cls_instance, algorithm, size, image_path):
    return cls_instance.work(algorithm, size, image_path)


class ImageProcess(object):
    def __init__(self):
        self.option_dict = self.get_options()
        # print(self.option_dict)

    # configure file
    def get_options(self):
        cf = ConfigParser.ConfigParser()

        if os.path.exists("config.cof"):
            cf.read('config.cof')
        else:
            print("there is no config.cof!")
            exit()

        option_dict = dict()
        for key, value in cf.items("MAIN"):
            option_dict[key] = eval(value)

        return option_dict

    # select algorithm
    def get_algorithm(self):
        algorithms = []
        for algorithm in self.option_dict["algorithm"]:
            if algorithm == "HOG":
                algorithms.append(my_hog.HOG())
            if algorithm == "LBP":
                algorithms.append(my_lbp.LBP())
            if algorithm == "GLCM":
                algorithms.append(my_glcm.GLCM())
            if algorithm == "HESSIAN":
                algorithms.append(my_hessian.HESSIAN())
            if algorithm == "CANNY":
                algorithms.append(my_canny.CANNY())

        return algorithms

    # image read function
    def image_read(self, algorithm):
        print("using single process to dealing with pictures.......\n")
        feature_list = []
        name_list = []

        # using the size of first image as default
        for file in os.listdir(self.option_dict["folder"]):
            if file.split(".")[-1] in self.option_dict["image_format"]:
                im = Image.open(os.path.join(self.option_dict[
                                "folder"], file)).convert("L")
                im_size = im.size
                break

        if self.option_dict["image_size"]:
            size = self.option_dict["image_size"]
            print("you set the image size as : {}\n".format(size))
        else:
            size = im_size
            print("using the first image's size {} as default\n\n".format(size))

        # start = time.time()

        for file in os.listdir(self.option_dict["folder"]):
            if file.split(".")[-1] in self.option_dict["image_format"]:
                print("processing: {}".format(file))
                name_list.append(file)
                feature = algorithm.read_image(os.path.join(
                    self.option_dict["folder"], file), size)
                feature_list.append(feature)

        # end = time.time()
        # print("using time: ".format(end - start))
        # exit()

        feature_list = np.array(feature_list)

        return feature_list, name_list

    def work(self, algorithm, size, image_path):

        feature = algorithm.read_image(image_path, size)
        image_name = os.path.split(image_path)[-1]

        return feature, image_name

    # multiprocessing the images
    def multiprocessing_read(self, algorithm):

        print("using multiprocesses to deal with pictures......\n")
        feature_list = []

        for file in os.listdir(self.option_dict["folder"]):
            if file.split(".")[-1] in self.option_dict["image_format"]:
                im = Image.open(os.path.join(self.option_dict[
                                "folder"], file)).convert("L")
                im_size = im.size
                break

        if self.option_dict["image_size"]:
            size = self.option_dict["image_size"]
            print("you set the image size as : {}\n".format(size))
        else:
            size = im_size
            print("using the first image's size {} as default\n".format(size))

        # start = time.time()

        worker = partial(proxy, self, algorithm, size)

        pool = multiprocessing.Pool(self.option_dict["njob"])

        image_path_list = [os.path.join(self.option_dict["folder"], name) for
                           name in os.listdir(self.option_dict["folder"]) if name.split(".")[-1]
                           in self.option_dict["image_format"]]

        result = pool.map(worker, image_path_list)

        # end = time.time()
        # print("using time: {}".format(end - start))
        # exit()

        feature_list = np.array([item[0] for item in result])
        name_list = np.array([item[1] for item in result])

        dataset = pd.DataFrame(data=feature_list, index=name_list,
                               columns=list(range(feature_list.shape[1])))

        return dataset,name_list


    # normalization
    def normalize(self, feature):
        normalizer = MinMaxScaler()
        normalized_feature = normalizer.fit_transform(feature)
        return normalized_feature

    # merge the features from different algorithms
    def merge_dataset(self):

        dataset_index = 0
        algorithm_list = self.get_algorithm()

        start = time.time()

        if self.option_dict["njob"] == 1:
            for algorithm in algorithm_list:
                print(algorithm)
                if dataset_index == 0:
                    left, name_list = self.image_read(algorithm)
                    if self.option_dict["normalize"]:
                        left = self.normalize(left)
                        # print(left[0:5])
                    dataset_index += 1
                else:
                    left = np.hstack((left, self.image_read(algorithm)[0]))
                    if self.option_dict["normalize"]:
                        left = self.normalize(left)
                        # print(left[0:5])

        elif self.option_dict["njob"] > 1:
            for algorithm in algorithm_list:
                dataset,name_list = self.multiprocessing_read(algorithm)
                if dataset_index == 0:
                    if self.option_dict["normalize"]:
                        left = self.normalize(dataset)
                        # print(left[0:5])
                    dataset_index += 1
                else:
                    left = np.hstack((left, self.multiprocessing_read(algorithm)[0]))
                    if self.option_dict["normalize"]:
                        left = self.normalize(left)
                        # print(left[0:5])
        else:
            print("you should write the true value of njob")

        end = time.time()
        
        print("using {} seconds to extract image feature ".format(end - start))    
        
        exit()    
        if self.option_dict["pca"]:
            left = my_pca.implement_pca(left)

        dataset = pd.DataFrame(data=left, index=name_list,
                               columns=list(range(left.shape[1])))
            
        return dataset

    

    def save_dataset(self, dataset):
        format_list = self.option_dict["save_format"]
        decimals = self.option_dict["decimals"]
        saving_name = os.path.split(self.option_dict["folder"])[-1]
        algorithm = "_".join(self.option_dict["algorithm"])

        if decimals:
            print("\nReserving {} significant digits....".format(decimals))
            dataset = dataset.round(decimals=decimals)

        if not os.path.exists("features"):
            os.mkdir("features")

        print("\nSaving features file......")
        if "csv" in format_list:
            dataset.to_csv("features/{}_{}.csv".format(saving_name, algorithm))

        if "excel" in format_list:
            dataset.to_csv(
                "features/{}_{}.xlsx".format(saving_name, algorithm))

        if "json" in format_list:
            dataset.to_json(
                "features/{}_{}.json".format(saving_name, algorithm))

        if "txt" in format_list:
            dataset.to_csv("features/{}_{}.txt".format(saving_name, algorithm))

        if "pickle" in format_list:
            dataset.to_pickle(
                "features/{}_{}.pkl".format(saving_name, algorithm))

        print("\nsuccessful!\n")

    def run(self):
        dataset = self.merge_dataset()
        print("\nGetting {} samples, every sample has {} features".format(
            dataset.shape[0], dataset.shape[1]))
        print(dataset.shape)   #"[sample,feature]"
        self.save_dataset(dataset)


if __name__ == '__main__':
    processor = ImageProcess()
    processor.run()
