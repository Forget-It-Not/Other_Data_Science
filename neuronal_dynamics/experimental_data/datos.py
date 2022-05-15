import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import gzip


#Each of this blocks is loading the .txt files and saving data into numpy arrays: tiempo, corriente and voltage

with open('tiempo.txt', 'rb') as f:
    u = pkl._Unpickler(f)
    u.encoding = 'latin1'
    tiempo = u.load()
    print(tiempo)

with open('current.txt', 'rb') as f:
    u = pkl._Unpickler(f)
    u.encoding = 'latin1'
    corriente = u.load()
    print(corriente)

with open('volt.txt', 'rb') as f:
    u = pkl._Unpickler(f)
    u.encoding = 'latin1'
    voltage = u.load()
    print(voltage)


#Plot of I-t and V-t just to check data has been correctly loaded
#<< Keep in mind the recording is very long, to handle we should slice a interval from the total data >>

plt.plot(tiempo, corriente)
plt.show()
plt.plot(tiempo, voltage)
plt.show()
