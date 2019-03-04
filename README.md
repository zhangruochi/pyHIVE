## This is the README file of PyHIVE

### How to configure the running environment.

1. you should download python2 or python3 from the offcial website _https://www.python.org_
2. you should install the dependent packages.
    - for the Linux and MacOS, we provide the requirements.txt. 
      > pip install -r requirements.txt

    - for the Windows, if you use the command above, you may not succeed. Two way for you to install the dependent packages.for the first,you can open the requirement.txt file, and install the pacakges one by one. But we really recommend you to use the second way. Installing the ANACONDA which include 1,000+ data science packages.You can download it from _https://www.continuum.io/downloads_

### How to run the example dataset in the paper

#### getting the HOG features.
1. enter in the foder of pyHIVE, open the configure file.
2. change the parameter of algorithm in [MAIN] module to ["HOG"], change the parameter of pca in [MAIN] module to True
3. run the main.py file, just type the command in command line tool. 
   > python main.py 
4. you can find the Img\_HOG.pkl file in the folder of feture. it is the features extracting from the images.

5. running the test.py to get the performace of the features in the command line tool

  > python test.py

#### getting the LBP features. just like above.

1. change the parameter of algorithm in [MAIN] module to ["LBP"].
2. all the other steps are the same as step 1.
3. Thirdly, getting the fusion feature of HOG and LBP.
4. change the parameter of algorithm in [MAIN] module to ["HOG","LBP"].
5. all the other steps are the same as above.