# Caveman
INTRO
Caveman is a Deep Learning project aming to convert ASL(American Sign Language) in to English Language with the help of CNN.

Discription 
This caveman uses CNN base nural network to Predict the gesture made. It incorporates OpenCV and Pillow library to get and display results and input.

Map:
data_collection.py : This python file is used to make dataset
image_processing.py : Contains code for fillter that are used in preposessing data
preprocessing.py : This is used to apply filtter on images and store those images so i can be latter used. This code also 
                   creates a Data Pipeline to be used by Deep learing model
train.py: This is used to train and store Deep Leaning model
pre.py : This is the final product of the entire project
          Press 1 to get prediction
          Press ESC to close the program
          the result is displayed in the terminal. with the number of letter as a result.
