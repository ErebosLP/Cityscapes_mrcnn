# Cityscapes_mrcnn
Hallo Giannmarco,
to run our Code u need to change some paths and maybe variabels as follows:
1. Path to the Data: In Line 264 in Instance_segmantation.py you need to define the rootpath for the dataset. it should look like this:
    - root 
        - leftImg8bit
            - train
            - val
        - gtFine
            - train
            - val
2. if u dont have the data on the lab computer here is the link for getting the dataset:https://www.cityscapes-dataset.com/downloads/
3. In line 221 u can define the number of epochs to train (currrently: 20) if the training is finished fast u may can go up with this so we get better results
4. If we the training goes really fsat you can also change the nuber of Pictures that is used for training. this can be defined by changing the return value of the __len__() function in line 34 in city_dataloader.py (currelntly: 100)

