import numpy as np
import pandas as pd #导入数据分析库Pandas
import Initial_NeuralNetwork as INN

inputfile = 'C:/Users/yzc/Desktop/1.xlsx' #销量数据路径
data = pd.read_excel(inputfile)

x = np.array([[1, 2, 3], [2, 3, 4]])
print(data)