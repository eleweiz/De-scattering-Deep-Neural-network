# De-scattering-Deep-Neural-network

There are 4 steps in total in this shared code, in which the last 2 steps are optional. In the “main_forward.m” file, there is a “Parameters definitions” section, where readers can replace with their own parameters if needed.

	Step 1. Generate training data from PSTPM stack using the file “main_forward.m”. The PSTPM stack are contained in the "PSTPM_data" folder, where others can put their own PSTPM data in the folder to generate training data.
As a toy example, we also offer one PSTPM stack, which can be downloaded from:
https://drive.google.com/drive/folders/1_r-yzPWGeYJhBtn1NxnjF937JjFXU02I?usp=sharing

	Step 2. Train the 3D neural network using the file "main_inverse.py"：Train the network with the data generated from Step 1. 

	Step 3 (Optional). Test and Save the test results with the file "Save_data.py". Remember to replace the trained model name. 

	Step 4 (Optional). Display your results with Matlab using the file "Results_Demo.m". As an example, we also offer " mScarlet-I" experimental results as an example in this step, which can be downloaded from:
https://drive.google.com/drive/folders/1_r-yzPWGeYJhBtn1NxnjF937JjFXU02I?usp=sharing

