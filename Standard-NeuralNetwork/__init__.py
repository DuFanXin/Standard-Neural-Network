import numpy as np
import Initial_NeuralNetwork as INN
l = np.array([3, 3, 2, 1])
w = []
w.clear()
#w.append(INN.initial_W(unitsNum_inLayer = l))
#w = INN.initial_W(unitsNum_inLayer = l)
w = np.append(w, INN.initial_W(unitsNum_inLayer = l))
print(w[0].shape)