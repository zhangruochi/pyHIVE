from PIL import Image
import os
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

from matplotlib import pyplot as plt
from io import BytesIO

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
    """a helper function for Parallel Computing
    """

    return cls_instance.work(algorithm, size, image_path)


def create_visualization_dir():
    """create the visualization directory
    """
    if not os.path.exists("visualization"):
        os.mkdir("visualization")
    else:
        import shutil
        shutil.rmtree("visualization")
        os.mkdir("visualization")


class ImageProcess(object):
    """
    the main class of pyHIVES    
    """

    def __init__(self):
        self.option_dict = self.get_options()
        # print(self.option_dict)

    def get_options(self):
        """read the configure file
           this function generate a dict，all the parameters are the items of dict 
        """
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

    def get_algorithm(self):
        """algorithm selecting function
           there are five different algorithms,
           this function select the algorithm 
           through the configure file.
        """
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

    def visualization(self,feature,file,algorithm):
        """draw the histogram
        """
        plt.figure(1,figsize = (8,6))
        plt.hist(feature,bins = 30)
        plt.savefig(os.path.join("visualization",file.split(".")[0] + "_" + algorithm.get_name()))
        plt.close()

    def image_read(self, algorithm):
        """image reading function
           input: the algorithm you have selected in the configure file
           output: feature_list contains all the features pyHIVES have 
           extracted, every image is converted into a vector, so all the 
           images are converted into a matrix.
           name_list conatians the name of features,the location 
           of names is corresponding to the location of features in 
           the feature_list 
        """
        print("using single process to dealing with pictures......\n")
        feature_list = []
        name_list = []
        im_size = None

        # using the size of first image as default
        for file in os.listdir(self.option_dict["folder"]):
            if file.split(".")[-1] in self.option_dict["image_format"]:
                im = Image.open(os.path.join(self.option_dict[
                                "folder"], file)).convert("L")
                im_size = im.size
                break

        if not im_size:
            print("can not find image in the image folder!\n")
            exit()

        if self.option_dict["image_size"]:
            size = self.option_dict["image_size"]
            print("you set the image size as : {}\n".format(size))
        else:
            size = im_size
            print("using the first image's size {} as default\n\n".format(size))

        # start = time.time()

        # list all the image and extracted them one by one
        for file in os.listdir(self.option_dict["folder"]):
            if file.split(".")[-1] in self.option_dict["image_format"]:
                print("processing: {}".format(file))
                name_list.append(file)
                feature = algorithm.read_image(os.path.join(
                    self.option_dict["folder"], file), size)
                if self.option_dict["visualization"]:
                    self.visualization(feature,file,algorithm)

                feature_list.append(feature)
                

                #print(file)
                #print(feature)
                #exit()


        # end = time.time()
        # print("using time: ".format(end - start))
        # exit()

        feature_list = np.array(feature_list)

        return feature_list, name_list

    def work(self, algorithm, size, image_path):
        """this function is a worker of Parallel Computing function(multiprocessing_read)
        input: the algorithm you have choosed, the size you want to converted 
        images into, the paths of images
        output: feature and the name of feature
        """
        feature = algorithm.read_image(image_path, size)
        image_name = os.path.split(image_path)[-1]
        


        return feature, image_name

    def multiprocessing_read(self, algorithm):
        """ images multiprocessing function, we created a 
            process pool，and the process in the pool will call 
            the worker function circularly untill all the tasks 
            are done.
        """

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

        for feature,file in zip(feature_list,name_list):
            self.visualization(feature,file,algorithm)


        dataset = pd.DataFrame(data=feature_list, index=name_list,
                               columns=list(range(feature_list.shape[1])))

        return dataset, name_list

    def normalize(self, feature):
        """normalize the features.
        """
        normalizer = MinMaxScaler()
        normalized_feature = normalizer.fit_transform(feature)
        return normalized_feature

    def merge_dataset(self):
        """merge the features from different algorithms.
        """
        dataset_index = 0
        algorithm_list = self.get_algorithm()

        # extracting features with single process
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

        # extracting features with multiple computing cores
        elif self.option_dict["njob"] > 1:
            for algorithm in algorithm_list:
                dataset, name_list = self.multiprocessing_read(algorithm)
                if dataset_index == 0:
                    if self.option_dict["normalize"]:
                        left = self.normalize(dataset)
                        # print(left[0:5])
                    dataset_index += 1
                else:
                    left = np.hstack(
                        (left, self.multiprocessing_read(algorithm)[0]))
                    if self.option_dict["normalize"]:
                        left = self.normalize(left)
                        # print(left[0:5])
        else:
            print("you should write the true value of njob")

        if self.option_dict["pca"]:
            left = my_pca.implement_pca(left)

        dataset = pd.DataFrame(data=left, index=name_list,
                               columns=list(range(left.shape[1])))

        return dataset

    def save_dataset(self, dataset):
        """saving the extracted features.
           input: DataFrame of features.
           output: one or more formats which are selected in configure file.
        """
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
        """ high-level function to run the entire class.
        """ 
        create_visualization_dir()
        dataset = self.merge_dataset()
        print("\nGetting {} samples, every sample has {} features".format(
            dataset.shape[0], dataset.shape[1]))
        # print(dataset.shape)   #"[sample,feature]"
        self.save_dataset(dataset)


if __name__ == '__main__':
    processor = ImageProcess()
    processor.run()
