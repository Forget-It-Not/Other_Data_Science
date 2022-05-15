import numpy as np
import pandas as pd
import scipy.stats as st

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection


### Carga de datos ###
#--------------------#
path = "data"
Cdata = pd.read_csv(f"{path}/TrozoC.txt", skiprows=3, header=None, sep="\t")
Gdata = pd.read_csv(f"{path}/TrozoG.txt", skiprows=3, header=None, sep="\t")
Rdata = pd.read_csv(f"{path}/TrozoR.txt", skiprows=3, header=None, sep="\t")


### Normalización ###
#-------------------#
Cdata = (Cdata - np.mean(Cdata)) / np.std(Cdata)
Gdata = (Gdata - np.mean(Gdata)) / np.std(Gdata)
Rdata = (Rdata - np.mean(Rdata)) / np.std(Rdata)


### Gráficas umbral binarización ###
#----------------------------------#
def thresh_plot(series, thresh):
    '''
    Dibuja una gráfica de <series> coloreando las partes por encima y por debajo
    del umbral <thresh>
    '''
    x = np.array(series.index)
    y = np.array(series)
    cmap = ListedColormap(['b', 'g', 'r'])
    norm = BoundaryNorm([-1, thresh, thresh, 1], cmap.N)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(series)

    plt.figure()
    plt.gca().add_collection(lc)
    plt.plot([x.min(), x.max()], [thresh, thresh], "--", color="red")
    plt.xlim(x.min(), x.max())
    #plt.ylim(-1.1, 1.1)
    plt.show()


### Binarización ###
#------------------#
def binarize(series, thresh):
    '''
    Devuelve la serie de datos binarizada de forma xt = 1 if xt <= <thresh>
    xt = 0 otherwise
    '''
    bin_series = series.copy(deep=True)
    bin_series[bin_series < thresh] = 0
    bin_series[bin_series != 0] = 1
    return bin_series


### Enventanado ###
#-----------------#
def window(series, dt):
    '''
    Devuelve la serie de datos agrupada en ventanas de longitud <dt> tal que el
    valor en cada ventana es 1 si algún xt = 1 y 0 otherwise

    También devuelve el numero de spikes resultante en cada vetana y el número
    de spikes que continuan desde una ventana anterior (spikes cortadas)
    '''
    win_starts = np.arange(0, len(series), dt)
    win_series = np.zeros(len(win_starts))
    num_spikes = np.zeros(len(win_starts))
    num_cuts = np.zeros(len(win_starts))

    for i in range(len(win_starts)):
        window = np.array(series[win_starts[i]: win_starts[i]+dt])
        win_series[i] = np.any(window)
        num_spikes[i] = np.sum(np.diff(window) == 1) + \
            (1 if window[0] == 1 else 0)
        num_cuts[i] = window[0]

    return win_series, num_spikes, num_cuts


### Cálculo de probabilidades (cod. binaria) ###
#----------------------------------------------#
def bpcalc(series):
    '''
    Devuelve las probabilidades asociadas a una serie de datos (o una lista de
    series de datos)

    * Serie de datos: probabilidades marginales de X=1
    * Lista de 2 series de datos: probabilidades conjuntas (X=..., Y=...)
    '''
    if type(series) in [list, tuple]:
        s1s2 = np.array([series[0], series[1]])
        s1s2 = pd.DataFrame(s1s2.T, columns=["s1", "s2"])
        Pjoint = s1s2.value_counts(normalize=True)
        Pjoint = Pjoint.to_numpy()
        Pjoint = Pjoint.reshape(2, 2).T
        return Pjoint
    else:
        return np.mean(series)


### Cálculo de entropías H(X) y H(X,Y) ###
#----------------------------------------#
def H(probs):
    '''
    Devuelve la entropía de una serie de datos marginal o conjunta en base a su
    vector / tabla de probabilidades marginales / conjuntas
    '''
    return -np.sum(np.nan_to_num(probs*np.log2(probs)))


def loopMI(P1, P2, Pjoint):
    '''
    Devuelve la MI de un par de series de datos calculada iterando para cada
    valor (para verificar los resultados de usar H(X) + H(Y) - H(X,Y))
    '''
    suma = 0
    for i in range(len(P1)):
        for j in range(len(P2)):
            suma += np.nan_to_num(Pjoint[i, j]
                                  * np.log2((Pjoint[i, j])/(P1[i]*P2[j])))
    return suma


### Codificacion en palabras ###
#------------------------------#
def word_code(series, word_size):
    '''
    Devuelve la serie de datos codificada en palabras de tamaño <word_size>; en
    formato array N X word_size donde cada fila es un patrón (x[w], x[w+1], ...)
    '''
    word_series = np.zeros((len(series) - word_size + 1, word_size))
    for i in range(len(series) - word_size + 1):
        word_series[i] = series[i:i+word_size]
    return word_series


### Cálculo de probabilidades (cod. ventana) ###
#----------------------------------------------#
def wpcalc(series):
    '''
    Devuelve las probabilidades asociadas a una serie de datos (o una lista de
    series de datos)

    * Serie de datos: probabilidades marginales de X=...
    * Lista de 2 series de datos: probabilidades conjuntas (X=..., Y=...)
    '''
    if type(series) in [list, tuple]:
        index = 2**np.arange(series[0].shape[1])
        probs = np.zeros((2**series[0].shape[1], 2**series[0].shape[1]))
        s1_indexes = np.sum(index*series[0], axis=1)
        s2_indexes = np.sum(index*series[1], axis=1)
        join_indexes = np.array([s1_indexes, s2_indexes])
        join_indexes, counts = np.unique(
            join_indexes, axis=1, return_counts=True)
        probs[join_indexes[0].astype(
            int), join_indexes[1].astype(int)] += counts
    else:
        index = 2**np.arange(series.shape[1])
        probs = np.zeros(2**series.shape[1])
        indexes = np.sum(index*series, axis=1)
        indexes, counts = np.unique(indexes, return_counts=True)
        probs[indexes.astype(int)] += counts
    return probs / np.sum(probs)


### Representación por SAX y OP ###
#---------------------------------#

### Transformación PAA (media en una ventana)

def paa(series, dt):
    '''
    Devuelve la serie de datos agrupada en ventanas de longitud <dt> tal que el
    valor en cada ventana es la media de valores en el intervalo
    '''
    paa_starts = np.arange(0, len(series), dt)
    paa_series = np.zeros(len(paa_starts))

    for i in range(len(paa_starts)):
        paa = np.array(series[paa_starts[i]: paa_starts[i]+dt])
        paa_series[i] = np.mean(paa)

    return paa_series

### Codificación por SAX


def sax_code(paa_series, n_vals, breaks=False):
    '''
    Devuelve la codificación SAX de una PAA con <n_vals> valores

    Se puede especificar los umbrales de cada valor con <breaks>:
        * 0 si paa[i] < breaks[0]
        * 1 si paa[i] < breaks[1]
        ...
    '''
    if breaks == False:
        probs = np.linspace(0, 1, n_vals+1)
        breaks = st.norm.ppf(probs)
    breaks = breaks.reshape(-1, 1)
    return np.argmax(paa_series < breaks, axis=0) - 1

### Codificación por OP (ordinal patterns)


def op_code(paa_series, order, eps):
    '''
    Devuelve la codificación por OP de una PAA con <order> orden
    (nº de puntos incluidos en el patrón)

    <eps> establece la tolerancia a la hora de determinar si 2
    valores son iguales o no
    '''
    paa_seq = word_code(paa_series, order)
    op_series = np.zeros((paa_seq.shape[0], order-1))
    for j in range(paa_seq.shape[1]-1):
        diff = paa_seq[:, j+1] - paa_seq[:, j]
        diff[np.abs(diff) < eps] = 0
        op_series[:, j] = np.sign(diff)
    return op_series

### Generador de colores para el plot de OP ###


def color_gen(k, cmap_name="hsv"):
    cmap = plt.get_cmap(cmap_name)
    color = [cmap(i/k) for i in range(k)]
    counter = 0
    while True:
        yield color[counter]
        counter = (counter+1) % (k)


def op_plot(op_series, xdat=[], cmap_name="hsv"):
    order = op_series.shape[1]+1
    cgen = color_gen(order, cmap_name=cmap_name)
    if len(xdat) == 0:
        xdat = np.arange(len(op_series)+order-1, dtype=float)
    for i in range(op_series.shape[0]):
        x = np.copy(xdat[i:i+order])
        y = [0]
        for j in range(order-1):
            y.append(y[-1]+op_series[i, j])
        x[0], x[-1] = x[0]+0.3, x[-1]-0.3
        y = np.array(y) - min(y) + 0.08*(i % (order-1))
        plt.plot(x, y, color=next(cgen), lw=2)
        plt.plot(x, y, ".", color="black")


def op_plot2(op_series, xdat=[], cmap_name="hsv"):
    order = op_series.shape[1]+1
    cgen = color_gen(order, cmap_name=cmap_name)
    if len(xdat) == 0:
        xdat = np.arange(len(op_series), dtype=float)
    xdiff = xdat[1] - xdat[0]
    for i in range(op_series.shape[0]):
        x = np.linspace(xdat[i]-0.4*xdiff, xdat[i]+0.4*xdiff, order)
        #print(x)
        y = [0]
        for j in range(order-1):
            y.append(y[-1]+op_series[i, j])
        y = np.array(y) - min(y)
        plt.plot([xdat[i]-0.5*xdiff, xdat[i]-0.5*xdiff],
                 [0, order-1], "--", color="darkgrey")
        plt.plot([xdat[i]+0.5*xdiff, xdat[i]+0.5*xdiff],
                 [0, order-1], "--", color="darkgrey")
        plt.plot(x, y, color=next(cgen), lw=2)
        plt.plot(x, y, ".", color="black")


def spcalc(series, k):
    '''
    Devuelve las probabilidades asociadas a una serie de datos (o una lista de
    series de datos)

    * Serie de datos: probabilidades marginales de X=...
    * Lista de 2 series de datos: probabilidades conjuntas (X=..., Y=...)
    '''
    if type(series) in [list, tuple]:
        index = k**np.arange(series[0].shape[1])
        probs = np.zeros((k**series[0].shape[1], k**series[0].shape[1]))
        s1_indexes = np.sum(index*series[0], axis=1)
        s2_indexes = np.sum(index*series[1], axis=1)
        join_indexes = np.array([s1_indexes, s2_indexes])
        join_indexes, counts = np.unique(
            join_indexes, axis=1, return_counts=True)
        probs[join_indexes[0].astype(
            int), join_indexes[1].astype(int)] += counts
    else:
        index = k**np.arange(series.shape[1])
        probs = np.zeros(k**series.shape[1])
        indexes = np.sum(index*series, axis=1)
        indexes, counts = np.unique(indexes, return_counts=True)
        probs[indexes.astype(int)] += counts
    return probs / np.sum(probs)
