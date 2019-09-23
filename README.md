1.	Install python enviroment and install the package in file requirement.txt

2.  	Run the requirement file using
-   	Pip3 install –r Requirement.txt

3.	Predict the image : 
-	File : visualize.py
-	Change the image directory in ‘img_path’  line 14
-	Change the model path in ‘model_dir’ line 15 < default in folder “trained_weights” >
-	Change saving image directory in ‘save_dir’ line 16
-	Run “python3 visualize.py” and the result image is in save_dir   

4.	Eval the accuracy :
-	File : eval.py
-	Change the image directory in ‘img_path’  line 29
-	Change the model path in ‘model_dir’ line 30 < default in folder “trained_weights” >
-	Change annotation path in ‘anno_path’ line 31
-	Run “python3 eval.py” and you can see the accuracy

Note: 

The Trained DataSet model is not attached in this zip as it occupies more memory so I upload in the google drive Kindly make use of the following drive to access it

https://drive.google.com/open?id=1bJ_6p6RPpvzRn6E_xoISs8Jos9ijiOgr

Once you download the dataset unet.hdf5 file place it in the trained_weights folder and then run the execution for the prediction. 
