# Cityscapes_mrcnn
Hallo Giannmarco,
to run our Code u need to change some paths and maybe variabels as follows:
1. Path to the Data: In Line 264 in Instance_segmantation you need to define the rootpath for the dataset. it should look like this:
    - root
          - leftImg8bit
              - train
              - val
          - gtFine
              - train
              - val
2. if u dont have the data on the lab computer here is the link for getting the dataset:https://www.cityscapes-dataset.com/downloads/
3. In line 221 u can define the nuber of epochs to train (currrentliy: 20)

