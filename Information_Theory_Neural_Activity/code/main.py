import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn
import os

from ti_utils import *


### DATA HANDLING ###
#===================#

C0 = Cdata[0]
C1 = Cdata[1]

G0 = Gdata[0]
G1 = Gdata[1]
#G2 = Gdata[2]   ¡INYECCION DE GABA¡

R0 = Rdata[0]
R1 = Rdata[1]

data_list = [C0, C1, G0, G1, R0, R1]
data_names = ["Neurona LP - Control",
              "Neurona VD - Control",
              "Neurona LP - Glutamato",
              "Neurona VD - Glutamato",
              "Neurona LP - Recuperación",
              "Neurona VD - Recuperación"]


### Z-NORMALIZATION ###
#=====================#

### Comparison plots ###

cmap = matplotlib.cm.get_cmap('magma')

## Not normalized
Cdata = pd.read_csv(f"{path}/TrozoC.txt", skiprows=3, header=None, sep="\t")
Gdata = pd.read_csv(f"{path}/TrozoG.txt", skiprows=3, header=None, sep="\t")
Rdata = pd.read_csv(f"{path}/TrozoR.txt", skiprows=3, header=None, sep="\t")

C0u = Cdata[0]
C1u = Cdata[1]

G0u = Gdata[0]
G1u = Gdata[1]
#G2 = Gdata[2]   ¡INYECCION DE GABA¡

R0u = Rdata[0]
R1u = Rdata[1]

plt.plot(np.arange(0, 5000-3000)/10,
         C0u[3000:5000], color=cmap(1/8), label="LP - Control", lw=0.7)
plt.plot(np.arange(0, 9000-7000)/10,
         C1u[7000:9000], color=cmap(2/8), label="VD - Control", lw=0.7)

plt.plot(np.arange(0, 3000-1000)/10,
         G0u[1000:3000], color=cmap(3/8), label="LP - Glutamato", lw=0.7)
plt.plot(np.arange(0, 2000)/10, G1u[:2000],
         color=cmap(4/8), label="VD - Glutamato", lw=0.7)
#plt.plot(G2[20000:22000])

plt.plot(np.arange(0, 2000)/10,
         R0u[:2000], color=cmap(5/8), label="LP - Recuperación", lw=0.7)
plt.plot(np.arange(0, 15000-13000)/10,
         R1u[13000:15000], color=cmap(6/8), label="VD - Glutamato", lw=0.7)

plt.legend()
plt.xlabel("$t$ (ms)")
plt.ylabel("$V(t)$ (mV?)")
plt.savefig("figures/unnorm_comp.eps", dpi=320, format="eps")

## Not normalized extra

plt.plot(np.arange(0, 5000-3000)/10, C0u[3000:5000], color=cmap(0/7), lw=0.7)
plt.plot([0, 200], [C0u[3000:5000].max(), C0u[3000:5000].max()],
         "--", lw=1.2, color="red", label="$V_{max}$")
plt.plot([0, 200], [C0u[3000:5000].min(), C0u[3000:5000].min()],
         "--", lw=1.2, color="green", label="$V_{min}$")
plt.legend()
plt.xlabel("$t$ (ms)")
plt.ylabel("$V(t)$ (mV?)")
plt.savefig("eps_figures/unnorm_comp_extra_c0.eps", dpi=320, format="eps")


cmap = plt.get_cmap("magma")
plt.plot(np.arange(0, 5000-3000)/10, C0u[3000:5000], color=cmap(1/12), lw=0.7)
plt.plot(np.arange(0, 9000-7000)/10,
         C1u[7000:9000], color=cmap(2/12), lw=0.7, label="Series temporales")

plt.plot(np.arange(0, 3000-1000)/10, G0u[1000:3000], color=cmap(3/12), lw=0.7)

plt.plot([0, 200], [C0u[3000:5000].max(), C0u[3000:5000].max()],
         "--", lw=1.2, color="red", label="$V_{max}$")
plt.plot([0, 200], [C1u[7000:9000].max(), C1u[7000:9000].max()],
         "--", lw=1.2, color="red")
plt.plot([0, 200], [G0u[1000:3000].max(), G0u[1000:3000].max()],
         "--", lw=1.2, color="red")
plt.plot([0, 200], [C0u[3000:5000].min(), C0u[3000:5000].min()],
         "--", lw=1.2, color="blue", label="$V_{min}$")
plt.plot([0, 200], [C1u[7000:9000].min(), C1u[7000:9000].min()],
         "--", lw=1.2, color="blue")
plt.plot([0, 200], [G0u[1000:3000].min(), G0u[1000:3000].min()],
         "--", lw=1.2, color="blue")

plt.plot(np.arange(0, 2000)/10,
         G1u[:2000], color=cmap(4/12), lw=0.7)
#plt.plot(G2[20000:22000])

plt.plot(np.arange(0, 2000)/10, R0u[:2000], color=cmap(5/12), lw=0.7)
plt.plot(np.arange(0, 15000-13000)/10,
         R1u[13000:15000], color=cmap(6/12), lw=0.7)

plt.plot([0, 200], [G1u[:2000].max(), G1u[:2000].max()],
         "--", lw=1.2, color="red")
plt.plot([0, 200], [R0u[:2000].max(), R0u[:2000].max()],
         "--", lw=1.2, color="red")
plt.plot([0, 200], [R1u[13000:15000].max(), R1u[13000:15000].max()],
         "--", lw=1.2, color="red")
plt.plot([0, 200], [G1u[:2000].min(), G1u[:2000].min()],
         "--", lw=1.2, color="blue")
plt.plot([0, 200], [R0u[:2000].min(), R0u[:2000].min()],
         "--", lw=1.2, color="blue")
plt.plot([0, 200], [R1u[13000:5000].min(), R1u[13000:5000].min()],
         "--", lw=1.2, color="blue")

plt.legend()
plt.xlabel("$t$ (ms)")
plt.ylabel("$V(t)$ (mV?)")
plt.savefig("eps_figures/unnorm_comp_extra.eps", dpi=320, format="eps")

## Normalized
plt.plot(np.arange(0, 5000-3000)/10,
         C0[3000:5000], color=cmap(1/8), label="LP - Control", lw=0.7)
plt.plot(np.arange(0, 9000-7000)/10,
         C1[7000:9000], color=cmap(2/8), label="VD - Control", lw=0.7)

plt.plot(np.arange(0, 3000-1000)/10,
         G0[1000:3000], color=cmap(3/8), label="LP - Glutamato", lw=0.7)
plt.plot(np.arange(0, 2000)/10, G1[:2000],
         color=cmap(4/8), label="VD - Glutamato", lw=0.7)
#plt.plot(G2[20000:22000])

plt.plot(np.arange(0, 2000)/10,
         R0[:2000], color=cmap(5/8), label="LP - Recuperación", lw=0.7)
plt.plot(np.arange(0, 15000-13000)/10,
         R1[13000:15000], color=cmap(6/8), label="VD - Glutamato", lw=0.7)

plt.legend()
plt.xlabel("$t$ (ms)")
plt.ylabel("$V_z(t)$ (mV?)")
plt.savefig("figures/norm_comp.eps", dpi=320, format="eps")

## Normalized extra
plt.plot(np.arange(0, 5000-3000)/10, C0[3000:5000], color=cmap(1/12), lw=0.7)
plt.plot(np.arange(0, 9000-7000)/10, C1[7000:9000], color=cmap(2/12), lw=0.7)

plt.plot(np.arange(0, 3000-1000)/10, G0[1000:3000], color=cmap(3/12), lw=0.7)
plt.plot(np.arange(0, 2000)/10,
         G1[:2000], color=cmap(4/12), lw=0.7,  label="Series temporales")
#plt.plot(G2[20000:22000])

plt.plot(np.arange(0, 2000)/10, R0[:2000], color=cmap(5/12), lw=0.7)
plt.plot(np.arange(0, 15000-13000)/10,
         R1[13000:15000], color=cmap(6/12), lw=0.7)

plt.plot([0, 200], [C0[3000:5000].max(), C0[3000:5000].max()],
         "--", lw=1.2, color="red", label="$V_{max}$")
plt.plot([0, 200], [C1[7000:9000].max(), C1[7000:9000].max()],
         "--", lw=1.2, color="red")
plt.plot([0, 200], [G0[1000:3000].max(), G0[1000:3000].max()],
         "--", lw=1.2, color="red")
plt.plot([0, 200], [G1[:2000].max(), G1[:2000].max()],
         "--", lw=1.2, color="red")
plt.plot([0, 200], [R0[:2000].max(), R0[:2000].max()],
         "--", lw=1.2, color="red")
plt.plot([0, 200], [R1[13000:15000].max(), R1[13000:15000].max()],
         "--", lw=1.2, color="red")
plt.plot([0, 200], [C0[3000:5000].min(), C0[3000:5000].min()],
         "--", lw=1.2, color="blue", label="$V_{min}$")
plt.plot([0, 200], [C1[7000:9000].min(), C1[7000:9000].min()],
         "--", lw=1.2, color="blue")
plt.plot([0, 200], [G0[1000:3000].min(), G0[1000:3000].min()],
         "--", lw=1.2, color="blue")
plt.plot([0, 200], [G1[:2000].min(), G1[:2000].min()],
         "--", lw=1.2, color="blue")
plt.plot([0, 200], [R0[:2000].min(), R0[:2000].min()],
         "--", lw=1.2, color="blue")
plt.plot([0, 200], [R1[13000:5000].min(), R1[13000:5000].min()],
         "--", lw=1.2, color="blue")

plt.legend()
plt.xlabel("$t$ (ms)")
plt.ylabel("$V_z(t)$ (mV?)")
plt.savefig("figures/norm_comp_extra.eps", dpi=320, format="eps")

### BINARIZATION THRESHOLD ###
#============================#

### Data plots ###

## Long stretch
for series in data_list:
    plt.figure()
    plt.plot(series[:15000])

## Medium stretch
plt.plot(C0[3000:5000])
plt.plot(C1[7000:9000])

plt.plot(G0[1000:3000])
plt.plot(G1[:2000])
#plt.plot(G2[20000:22000])

plt.plot(R0[:2000])
plt.plot(R1[13000:15000])

plt.legend()
plt.xlabel("Tiempo (ms)")
plt.ylabel("Voltaje normalizado")
plt.savefig("figures/norm_comp.png", dpi=320)


## Single spike
plt.plot(C0[3170:3300])
plt.plot(C1[7170:7300])

plt.plot(G0[1100:1250])
plt.plot(G1[150:250])
#plt.plot(G2[20250:20500])

plt.plot(R0[:125])
plt.plot(R1[13250:13500])


### Histogram of values ###

## Vanilla histogram
for series in data_list:
    plt.figure()
    plt.hist(series, bins=np.linspace(-7, 10, 100))

## Logarithmic histogram
for name, series in zip(data_names, data_list):
    plt.figure()
    plt.hist(series, bins=np.linspace(-7, 10, 100), log=True)
    plt.plot([4, 4], [1e7, 1e0], "--", color='red')
    plt.xlabel("Voltaje normalizado")
    plt.ylabel("Número de puntos")
    plt.title(name)
    plt.savefig(f"figures/{name}.png", dpi=320)

## Logarithmic KDE
for name, series in zip(data_names, data_list):
    plt.figure()
    seaborn.kdeplot(series[:40000], log_scale=[False, True])
    plt.plot([4, 4], [1e-10 if name
             == "Neurona LP - Control" else 1e-6, 1e0], "--", color='red')
    plt.xlabel("$V_z(t)$")
    plt.ylabel("Densidad de probabilidad")
    plt.title(name)
    plt.savefig(f"figures/{name}.eps", dpi=320, format="eps")


### Binarization plots ###

def thresh_plot(series, thresh, title=None):
    x = np.array(series.index)
    plt.figure()
    plt.plot(x/10, series)
    plt.plot([(x/10).min(), (x/10).max()], [thresh, thresh], "--", color="red")
    plt.xlabel("$t$ (ms)")
    plt.ylabel("$V_z(t)$ (mV?)")
    plt.title(title)


## According to histograms 5 would be the best across all series (??)
thresh_plot(C0[3000:5000], 5)
thresh_plot(C1[7000:9000], 5)

thresh_plot(G0[1000:3000], 5)
thresh_plot(G1[:2000], 5)
#thresh_plot(G2[20000:22000], 5)

thresh_plot(R0[:2000], 5)
thresh_plot(R1[13000:15000], 5)

## But 4 maybe works a bit better
thresh_plot(C0[3000:5000], 4, title="Neurona LP - Control")
plt.savefig("figures/C0_t4.eps", dpi=320, format="eps")

thresh_plot(C1[7000:9000], 4, title="Neurona VD - Control")
plt.savefig("figures/C1_t4.eps", dpi=320, format="eps")

thresh_plot(G0[1000:3000], 4, title="Neurona LP - Glutamato")
plt.savefig("figures/G0_t4.eps", dpi=320, format="eps")

thresh_plot(G1[:2000], 4, title="Neurona VD - Glutamato")
plt.savefig("figures/G1_t4.eps", dpi=320, format="eps")

#thresh_plot(G2[20000:22000], 5)

thresh_plot(R0[:2000], 4, title="Neurona LP - Recuperación")
plt.savefig("figures/R0_t4.eps", dpi=320, format="eps")

thresh_plot(R1[13000:15000], 4, title="Neurona VD - Recuperación")
plt.savefig("figures/R1_t4.eps", dpi=320, format="eps")


### BINARIZATION OF DATA ###
#==========================#

bdata_list = []
for serie in data_list:
    bdata_list.append(binarize(serie, 4))

C0b, C1b, G0b, G1b, R0b, R1b = bdata_list


### WINDOWING OF DATA ###
#=======================#

### Multiple spikes in each window with large window_size ###

multiple_spikes = []
for window_size in np.arange(50, 500, 50):
    print(window_size)
    _, num_spikes, _ = window(R1b, window_size)
    multiple_spikes.append(np.mean(num_spikes > 1))

plt.plot(np.arange(50, 500, 50)/10, multiple_spikes)
plt.xlabel("$\Delta t$ (ms)")
plt.ylabel("Frecuencia de spikes múltiples")
plt.savefig("figures/window_multiple_spikes_R1.eps", dpi=320, format="eps")

### Spikes cut across several windows with small window_size ###

cut_spikes = []
for window_size in np.arange(10, 160, 20):
    print(window_size)
    _, _, num_cut = window(R0b, window_size)
    cut_spikes.append(np.mean(num_cut))

plt.plot(np.arange(10, 160, 20)/10, cut_spikes)
plt.xlabel("$\Delta t$ (ms)")
plt.ylabel("Frecuencia de spikes incompletos")
plt.ylim(0, 0.012)
plt.title("Neurona LP - Recuperación")
plt.savefig("figures/window_cut_spikes_R0.eps", dpi=320, format="eps")


### PAA TRANSFORMATION ###
#========================#

### PAA initial testing ###

## PAA conversion
C0paa = paa(C0[3170:3300], 10)

plt.plot(np.arange(3170, 3300)/10, C0[3170:3300], label="Voltaje")
plt.plot(np.linspace(3170, 3300, len(C0paa))/10, C0paa, "o--", color="red")

### PAA conversion of series ###

dt = 7
paa_data_list = []
for data, name in zip(data_list, data_names):
    print(name)
    paa_data = paa(data, dt)
    paa_data_list.append(paa_data)
print("over")

filenames = [f'results/paa_dt{dt}_{s}.csv' for s in ['C0', 'C1', 'G0', 'G1', 'R0', 'R1']]
for paa_data, filename in zip(paa_data_list, filenames):
    pd.DataFrame(paa_data).to_csv(filename, index=False)

paa_data_list = []
for filename in filenames:
    paa_data_list.append(np.loadtxt(filename, skiprows=1))

# np.loadtxt('results/paa_dt7_C0.csv',skiprows=1)

### SAX CODIFICATION ###
#======================#

### SAX initial testing

C0sax = sax_code(C0paa, 4)

plt.plot(C0sax)

### SAX conversion of series ###

for k in range(3,4+1):
    print(k)
    sax_data_list = []
    for data, name in zip(paa_data_list, data_names):
        sax_data = sax_code(data, k)
        sax_data_list.append(sax_data)

    filenames = [f'results/sax_dt{dt}_k{k}_{s}.csv' for s in ['C0', 'C1', 'G0', 'G1', 'R0', 'R1']]
    for sax_data, filename in zip(sax_data_list, filenames):
        pd.DataFrame(sax_data).to_csv(filename, index=False)
print("over")

### SAX wording ###

for file in os.listdir(path='results'):
    key = file.split("_")[0]
    if key == 'sax':
        print(file)
        sax_data = np.loadtxt('results/'+file, skiprows=1)
        for word_size in range(2,8+1):
            print(word_size)
            sax_data_worded = word_code(sax_data, word_size)
            outfile = ('results/' + file).replace('w1', f'w{word_size}')
            pd.DataFrame(sax_data_worded).to_csv(outfile, index=False)
print("over")

### OP CODIFICATION ###
#=====================#

### OP initial testing

C0op = op_code(C0paa, 3, 0.5)

op_plot(C0op)

op_plot2(C0op)
plt.xlabel("$t$ (ms)")
plt.ylabel("$V_{OP}^*(t)$")
plt.title("$V_{OP}^*(t)$, $\Delta t="+str(10)+"$, $k=3$")

### OP conversion of series ###

for k in range(5,8+1):
    print(k)
    op_data_list = []
    for data, name in zip(paa_data_list, data_names):
        op_data = op_code(data, k, 0.5)
        op_data_list.append(op_data)

    filenames = [f'results/op_dt{dt}_k{k}_{s}.csv' for s in ['C0', 'C1', 'G0', 'G1', 'R0', 'R1']]
    for op_data, filename in zip(op_data_list, filenames):
        pd.DataFrame(op_data).to_csv(filename, index=False)
print("fin")

### PAA, SAX and OP decision plots ###

for dt in [3, 5, 7, 10, 15, 30]:
    paa_data = paa(C0[3170:3300], dt)
    # plt.figure()
    # plt.plot(np.arange(3170, 3300)/10, C0[3170:3300], label="$V_z(t)$")
    # plt.plot(np.linspace(3170, 3300, len(paa_data))/10, paa_data, "o--",
    #          color="red", label="$V_{PAA}^*(t)$, $\Delta t="+str(dt)+"$")
    # plt.xlabel("$t$ (ms)")
    # plt.ylabel("$V_z(t)$")
    # plt.title(f"$dt={dt}$")
    # plt.legend(loc='upper right')
    # plt.savefig(f"figures/C0_PAA_dt{dt}.eps", dpi=320, format="eps")

    # ## SAX k=3
    # sax_data = sax_code(paa_data, 3)
    # plt.figure()
    # #plt.plot(np.arange(3170, 3300)/10, C0[3170:3300], label="Voltaje")
    # plt.plot([317, 330], [0, 0], "--", color="grey")
    # plt.plot([317, 330], [1, 1], "--", color="grey")
    # plt.plot([317, 330], [2, 2], "--", color="grey")
    # plt.plot(np.linspace(3170, 3300, len(sax_data))/10, sax_data,
    #          color="red", label="$V_{SAX}^*(t)$, $\Delta t="+str(dt)+"$, $k=3$")
    # plt.xlabel("$t$ (ms)")
    # plt.ylabel("$V_{SAX}^*(t)$")
    # plt.title(f"$dt={dt}$")
    # plt.legend(loc='upper right')
    # plt.savefig(f"figures/C0_SAX_dt{dt}_k3.eps", dpi=320, format="eps")
    #
    # ## SAX k=4
    # sax_data = sax_code(paa_data, 4)
    # plt.figure()
    # #plt.plot(np.arange(3170, 3300)/10, C0[3170:3300], label="Voltaje")
    # plt.plot([317, 330], [0, 0], "--", color="grey")
    # plt.plot([317, 330], [1, 1], "--", color="grey")
    # plt.plot([317, 330], [2, 2], "--", color="grey")
    # plt.plot([317, 330], [3, 3], "--", color="grey")
    # plt.plot(np.linspace(3170, 3300, len(sax_data))/10, sax_data,
    #          color="red", label="$V_{SAX}^*(t)$, $\Delta t="+str(dt)+"$, $k=4$")
    # plt.xlabel("$t$ (ms)")
    # plt.ylabel("$V_{SAX}^*(t)$")
    # plt.title(f"$dt={dt}$")
    # plt.legend(loc='upper right')
    # plt.savefig(f"figures/C0_SAX_dt{dt}_k4.eps", dpi=320, format="eps")

    ## OP k=3
    op_data = op_code(paa_data, 3, 0.5)
    plt.figure()
    op_plot2(op_data, xdat=np.linspace(3170, 3300, len(op_data)+2)/10)
    plt.xlabel("$t$ (ms)")
    plt.ylabel("$V_{OP}^*(t)$")
    plt.title("$V_{OP}^*(t)$, $\Delta t="+str(dt)+"$, $k=3$")
    #plt.legend(loc='upper right')
    plt.savefig(f"figures/C0_OP2_dt{dt}_k3.eps", dpi=320, format="eps")

    ## OP k=4
    op_data = op_code(paa_data, 4, 0.5)
    plt.figure()
    op_plot2(op_data, xdat=np.linspace(3170, 3300, len(op_data)+3)/10)
    plt.xlabel("$t$ (ms)")
    plt.ylabel("$V_{OP}^*(t)$")
    plt.title("$V_{OP}^*(t)$, $\Delta t="+str(dt)+"$, $k=4$")
    #plt.legend(loc='upper right')
    plt.savefig(f"figures/C0_OP2_dt{dt}_k4.eps", dpi=320, format="eps")


### MI CALCULATIONS ###
#=====================#

### Binarized data (binarized_mi_calculations.ipynb)

### SAX

Hs = []
MI = []

for file in os.listdir("results"):
    keys = file.split("_")
    if keys[0] == "sax" and keys[1] == "dt7" and keys[3][-1] == '0' and keys[2][1:] != "4":
        keys2 = keys.copy()
        keys2[3] = keys2[3][:1]+'1'+keys2[3][2:]
        file2 = "_".join(keys2)
        print(file, file2)

        data1 = np.loadtxt("results/"+file, usecols=range(int(keys[4][1:-4])), skiprows=1, delimiter=",")
        data2 = np.loadtxt("results/"+file2, usecols=range(int(keys[4][1:-4])), skiprows=1, delimiter=",")
        if keys[4] == 'w1.csv':
            data1 = data1.reshape(-1,1)
            data2 = data2.reshape(-1,1)

        PC0 = spcalc(data1, int(keys[2][1:]))
        PC1 = spcalc(data2, int(keys[2][1:]))

        Pjoint = spcalc([data1, data2], int(keys[2][1:]))

        H0 = H(PC0)
        H1 = H(PC1)
        MIcalc = H0 + H1 - H(Pjoint)

        line1 = {'dt': int(keys[1][2:]), 'k': int(keys[2][1:]), 'w':int(keys[4][1:2]),  'serie':keys[3][:1] , 'MI': MIcalc}
        line2 = {'dt': int(keys[1][2:]), 'k': int(keys[2][1:]), 'w':int(keys[4][1:2]),  'serie': keys[3][:1], 'neuron': 0, 'H': H0}
        line3 = {'dt': int(keys[1][2:]), 'k': int(keys[2][1:]), 'w':int(keys[4][1:2]), 'serie': keys[3][:1], 'neuron': 1, 'H': H1}
        MI.append(line1)
        Hs.append(line2)
        Hs.append(line3)
print("fin")

MIdata = pd.DataFrame(MI)
Hdata = pd.DataFrame(Hs)

### OP

MI = []
Hs = []

for file in os.listdir("results"):
    keys = file.split("_")
    if keys[0] == "op" and keys[3][1:2] == '0' and keys[2][1:] == "3":
        keys2 = keys.copy()
        keys2[3] = keys2[3][:1]+'1'+keys2[3][2:]
        file2 = "_".join(keys2)
        print(file, file2)

        data1 = np.loadtxt("results/"+file, usecols=range(int(keys[2][1:])-1), skiprows=1, delimiter=",")
        data2 = np.loadtxt("results/"+file2, usecols=range(int(keys[2][1:])-1), skiprows=1, delimiter=",")

        data1 = data1 + 1
        data2 = data2 + 1

        PC0 = spcalc(data1, 3)
        PC1 = spcalc(data2, 3)

        Pjoint = spcalc([data1, data2], 3)

        H0 = H(PC0)
        H1 = H(PC1)

        MIcalc = H0 + H1 - H(Pjoint)

        line1 = {'dt': int(keys[1][2:]), 'k': int(keys[2][1:]), 'serie':keys[3][:1] , 'MI': MIcalc}
        line2 = {'dt': int(keys[1][2:]), 'k': int(keys[2][1:]), 'serie': keys[3][:1], 'neuron': 0, 'H': H0}
        line3 = {'dt': int(keys[1][2:]), 'k': int(keys[2][1:]), 'serie': keys[3][:1], 'neuron': 1, 'H': H1}
        MI.append(line1)
        Hs.append(line2)
        Hs.append(line3)
print("fin")

MIdata = pd.DataFrame(MI)
Hdata = pd.DataFrame(Hs)
