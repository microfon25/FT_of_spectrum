# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt


def sr_delta(a):
    s = 0
    for i in range(len(a) - 1):
        s += abs(a[i] - a[i + 1])
    return s / len(a) - 1


def nm_to_Hz(a):
    return 299792458 / (a * 10 ** (-9))


def interpolation(x1, y1, x2, y2, x):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


with open("spectrum1.txt", "r", encoding="utf-8") as f:
    ExpData = [x[:-1].split("\t") for x in f.readlines()]

ExpData = [[float(y) for y in x] for x in ExpData]

# wavelength = [z[0] for z in ExpData]
wavelength = [z for z in range(500, 1500)]
frequency = [nm_to_Hz(z) for z in wavelength]
# amplitude = [z[1] if z[1] > 0.05 else 0. for z in ExpData]
# amplitude = [z[1] for z in ExpData]
amplitude = [1 if 900 <= z <= 950 else 0 for z in range(500, 1500)]

dwavelength = [wavelength[i + 1] - wavelength[i] for i in range(len(wavelength) - 1)]
dfrequency = [abs(frequency[i] - frequency[i + 1]) for i in range(len(frequency) - 1)]
ddfrequency = [abs(dfrequency[i] - dfrequency[i + 1]) for i in range(len(dfrequency) - 1)]

# print([dfrequency[i]/min(ddfrequency) for i in range(len(dfrequency)-1)])

dfreq = min(ddfrequency) if min(ddfrequency) < min(dfrequency) else min(dfrequency) / 9.99
# print("dfreq = ", dfreq)
# print(ddfrequency)
# print([dfrequency[i]/dfreq for i in range(len(dfrequency))])

frequency = list(reversed(frequency))
frequency1 = frequency.copy()
amplitude = list(reversed(amplitude))
amplitude1 = amplitude.copy()
# print("frequency = ", frequency , "/nfrequency1 = ", frequency1)
j = 0
k = 0
low_freq = frequency1[0]
print("простите, я думаю")
for i in range(len(frequency) - 1):
    up_freq = frequency[i + 1]
    # print("i = ", i , " i + j = " , i+j , " up_freq = " , up_freq)
    # print(" frequency[i+1] = " , frequency[i+1])
    k += j
    j = 0
    while low_freq + dfreq < up_freq:
        j = j + 1
        low_freq = frequency[i] + dfreq * j
        frequency1.insert(i + j + k, low_freq)
        amplitude1.insert(i + j + k, 0)
        amplitude1[i + j + k] = interpolation(frequency1[i + j + k - 1], amplitude1[i + j + k - 1],
                                              frequency1[i + j + k + 1], amplitude1[i + j + k + 1],
                                              frequency1[i + j + k])
        # print("i =", i, ", j =", j, ", k =", k, ", i+j+k =", i + j + k, ", i+j+k+1 =", i + j + k + 1, ", frequency1[i+j+k] =", frequency1[i + j + k], ", frequency1[i+j+k+1] =", frequency1[i + j + k + 1], ", amplitude1[i+j+k-1] =", amplitude1[i + j + k - 1], ", amplitude1[i+j+k] =", amplitude1[i + j + k], ", amplitude1[i+j+k+1] =", amplitude1[i + j + k + 1], ", len(frequency1) =", len(frequency1), ", len(amplitude1) =", len(amplitude1), ", low_freq =", low_freq, ", frequency[i+1] =", frequency[i + 1])

print("все еще думаю....")
# for i in range(len(frequency1)-1):
#     if frequency1[i] < frequency1[i+1]:
#         print(i, frequency1[i], frequency , "ok")
#     else: print(i, frequency1[i] , frequency , "ne ok")

# for i in range(len(frequency1)):
#     if frequency1[i] == frequency[0] or frequency1[i] == frequency[1] or frequency1[i] == frequency[2] or frequency1[i] == frequency[3] or frequency1[i] == frequency[4]: print("ravno", frequency1[i])
#     if amplitude1[i] == amplitude[0] or amplitude1[i] == amplitude[1] or amplitude1[i] == amplitude[2] or amplitude1[i] == amplitude[3] or amplitude1[i] == amplitude[4]: print("ravno", amplitude1[i])

# print(interpolation(1,2,3,4,2))
#
#
#


frequency_width = abs(frequency1[len(frequency1) - 1] - frequency1[0])
sample_rate = 1 / sr_delta(frequency1)

# fig1 = plt.figure(1)
# fig1.suptitle('Spectrum')
# plt.plot(wavelength, amplitude)
#
# fig10 = plt.figure(10)
# fig10.suptitle('Spectrum')
# plt.plot(frequency1, amplitude1)
#
# fig2 = plt.figure(2)
# fig2.suptitle('frequency')
# plt.plot(range(len(frequency)), frequency)
#
# fig20 = plt.figure(20)
# fig20.suptitle('wavelength')
# plt.plot(range(len(wavelength)), wavelength)
#
# fig3 = plt.figure(3)
# fig3.suptitle('dfrequency')
# plt.plot(range(len(dfrequency)), dfrequency)
#
# fig30 = plt.figure(30)
# fig20.suptitle('dwavelength')
# plt.plot(range(len(dwavelength)), dwavelength)
#
# fig4 = plt.figure(4)
# fig4.suptitle('ddfrequency')
# plt.plot(range(len(ddfrequency)), ddfrequency)

# print(299792458/(3500*10**(-9))/(1.*10**14))

N = int(np.round(sample_rate * frequency_width, 0))

# 100 3500 856542680043.7177 3400 3400 1.1674841467899303e-12 2912245112152043.0 3400.0000000039727 2912245112148640.0 1.1674841467899303e-12
# 500 1500 399589947353.2363 1000 1000 2.5025654589753803e-12 399589947354236.06 1000.0000000025018 399589947353236.4 2.5025654589753803e-12
# print(dx, N, len(x), sample_rate, dx, np.abs(x[0]-x[1]), np.abs(x[len(x)-1]-x[len(x)-2]), duration/10**14, sample_rate*duration, N*1/sample_rate, N/(N*1/sample_rate))

print("все хорошо, пожалуйста потерпите еще чуть-чуть") if len(frequency1) == len(amplitude1) else print("что-то не так...")
# print(len(frequency1), len(amplitude1))
yff = fft(amplitude1)
xf = fftfreq(N, 1 / sample_rate)[:N // 2]
xf = [z * 10 ** 15 for z in xf]
yf = 2.0 / N * np.absolute(yff[:N // 2])
# print(len(xf), len(yf))
yf = [float(i) for i in yf]

print("и теперь думаю...")
y1 = 0
y2 = 0
n1 = 0
n2 = 0
n_max = 0
print(n_max, len(yf), max(yf) / 2, [n2, y2], [n1, y2])
print(yf[0:100])
n_max = yf.index(max(yf))
for n in range(len(yf) - 1):
    # if max(yf) == yf[n]:
    #     n_max = n
        # print("вы в 1 ife", n_max, len(yf), max(yf) / 2, [n + 1, yf[n + 1]], [n, yf[n]], [n2, y2], [n1, y2])
    if yf[n + 1] <= max(yf) / 2 <= yf[n]:
        n1 = n
        y1 = yf[n]
        n2 = n + 1
        y2 = yf[n + 1]
        break
        # print("вы в 2 ife", n_max, len(yf), max(yf) / 2, [n+1, yf[n+1]], [n , yf[n]], [n2, y2], [n1, y2])
    # print("вы в fore", n_max, len(yf),  max(yf) / 2, [n+1, yf[n+1]], [n , yf[n]], [n2, y2], [n1, y2])

print(max(yf) == yf[n_max], n_max, max(yf), yf[n_max], max(yf) / 2, len(yf), [n2, y2], [n1, y2])

dlit = (xf[n2] * (yf[n1] - (yf[n_max] / 2)) + xf[n1] * ((yf[n_max] / 2) - yf[n2])) / (yf[n1] - yf[n2]) * 2

# print("------------------------------------")
print("Длительность Авто-Корреляционной функции спектрально-ограниченного импульса = ", dlit, " фс")

# fig2 = plt.figure(2)
# fig2.suptitle('Fourier Transform - ACF Duration')
# ax = fig2.add_subplot()
# ax.set_title('FHWA ACF duration = ' + str(dlit) + ' fs')
# plt.plot(xf, yf)
# plt.grid()
#
# plt.show()

#
# import numpy as np
# from matplotlib import pyplot as plt
#
# SAMPLE_RATE = 44  # Гц
# DURATION = 100  # Секунды
#
# def generate_sine_wave(freq, sample_rate, duration):
#     x = np.linspace(0, duration, sample_rate*duration, endpoint=False)
#     frequencies = x * freq
#     # 2pi для преобразования в радианы
#     y = [1 if 3<=z<=5 else 0 for z in x]
#     return x, y
#
# # Генерируем волну с частотой 2 Гц, которая длится 5 секунд
# x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)
#
# plt.figure(1)
# plt.plot(x, y)
# # Number of samples in normalized_tone
# N = SAMPLE_RATE * DURATION
#
# yf = fft(y)
# xf = fftfreq(N, 1 / SAMPLE_RATE)
#
# plt.figure(2)
# plt.plot(xf, np.abs(yf))
# plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
