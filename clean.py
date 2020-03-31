import os 
import pandas as pd 
import shutil as sh


dataframe = pd.read_csv(os.path.join(os.getcwd(),'metadata.csv'))

img_path = os.path.join(os.getcwd(),'images')

path_pos = os.path.join(os.getcwd(),'positive_samples')


# print(dataframe.head())

# for ind,row in dataframe.iterrows():
# 	print(row['view'],row['filename'])
files = []
for ind,row in dataframe.iterrows():
	
	if row['view'] == 'PA' or row['view'] == 'AP'  :
		files.append(row['filename'])

for ind,file in enumerate(files):
	dest = os.path.join(path_pos , str(ind) +'.jpg')
	source = os.path.join(img_path , file)
	sh.copy(source,dest)
	