# Exploring Wavelet Analysis Tools in Python

This is brought about by the deprecation of `scipy.signal.cwt` in `scipy` v1.12 and its removal in v1.15. The `scipy` team recommends the use of `pywavelets`, so can we reproduce `scipy` wavelet analysis with `pywavelets` and other tools?

See:
- [`scipy.signal.cwt` documentation](https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.signal.cwt.html)
- [`pywavelets` documentation](https://pywavelets.readthedocs.io/en/latest/)

## Imports

We will need
- `numpy` for basic numerical operations
- `matplotlib.pyplot` for plotting
- `scipy.signal`<1.15 for the scipy implementation
- `pywt` for the pywavelets implementation


```python
%pip install -r requirements.txt
```

    Requirement already satisfied: numpy in /opt/conda/envs/notebook/lib/python3.12/site-packages (from -r requirements.txt (line 1)) (1.26.4)
    Collecting scipy<1.15 (from -r requirements.txt (line 2))
      Downloading scipy-1.14.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (60 kB)
    Requirement already satisfied: matplotlib in /opt/conda/envs/notebook/lib/python3.12/site-packages (from -r requirements.txt (line 3)) (3.10.1)
    Requirement already satisfied: pywavelets in /opt/conda/envs/notebook/lib/python3.12/site-packages (from -r requirements.txt (line 4)) (1.8.0)
    Requirement already satisfied: contourpy>=1.0.1 in /opt/conda/envs/notebook/lib/python3.12/site-packages (from matplotlib->-r requirements.txt (line 3)) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/notebook/lib/python3.12/site-packages (from matplotlib->-r requirements.txt (line 3)) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /opt/conda/envs/notebook/lib/python3.12/site-packages (from matplotlib->-r requirements.txt (line 3)) (4.56.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in /opt/conda/envs/notebook/lib/python3.12/site-packages (from matplotlib->-r requirements.txt (line 3)) (1.4.8)
    Requirement already satisfied: packaging>=20.0 in /opt/conda/envs/notebook/lib/python3.12/site-packages (from matplotlib->-r requirements.txt (line 3)) (24.2)
    Requirement already satisfied: pillow>=8 in /opt/conda/envs/notebook/lib/python3.12/site-packages (from matplotlib->-r requirements.txt (line 3)) (11.1.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /opt/conda/envs/notebook/lib/python3.12/site-packages (from matplotlib->-r requirements.txt (line 3)) (3.2.1)
    Requirement already satisfied: python-dateutil>=2.7 in /opt/conda/envs/notebook/lib/python3.12/site-packages (from matplotlib->-r requirements.txt (line 3)) (2.9.0.post0)
    Requirement already satisfied: six>=1.5 in /opt/conda/envs/notebook/lib/python3.12/site-packages (from python-dateutil>=2.7->matplotlib->-r requirements.txt (line 3)) (1.17.0)
    Downloading scipy-1.14.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (40.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m40.8/40.8 MB[0m [31m142.1 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hInstalling collected packages: scipy
      Attempting uninstall: scipy
        Found existing installation: scipy 1.15.2
        Uninstalling scipy-1.15.2:
          Successfully uninstalled scipy-1.15.2
    Successfully installed scipy-1.14.1
    Note: you may need to restart the kernel to use updated packages.



```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
```

## Set Up Signals

Let's look at two different kinds of signals:
1. a "chirp" signal, where the frequency changes with time
2. a multimodal sinusoid, where each sine component has equal amplitude


```python
def gaussian(x, x0, sigma):
    return np.exp(-((x - x0)/sigma)**2 / 2)


def make_chirp(t, t0, a):
    frequency = (a * (t + t0)) ** 2
    chirp = np.sin(2*np.pi*frequency*t)
    return chirp, frequency
    

def chirps(time):
    chirp1, frequency1 = make_chirp(time, 0.2, 9)
    chirp2, frequency2 = make_chirp(time, 0.1, 5)
    chirp = chirp1 + 0.6*chirp2
    chirp *= gaussian(time, 0.5, 0.2)
    return chirp, frequency1, frequency2


def sines(time, periods):
    y = np.sum([np.sin(2*np.pi*time/p) for p in periods], axis=0)
    return y

    
# generate chirp signal
time = np.linspace(0, 1.023, 2048)
chirp_signal, frequency1, frequency2 = chirps(time)

# plot signal
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(time, chirp_signal)
axs[1].plot(time, frequency1)
axs[1].plot(time, frequency2)
axs[1].set_yscale("log")
axs[1].set_xlabel("Time (s)")
axs[0].set_ylabel("Signal")
axs[1].set_ylabel("Frequency (Hz)")
plt.suptitle("Chirp signal")
plt.show()


# generate sines signal
periods = np.array([1/256, 1/64, 1/16, 1/4])
sine_signal = sines(time, periods)

# plot signal
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(time, sine_signal)
axs[1].hlines(1/periods, xmin=0, xmax=1)
axs[1].set_yscale("log")
axs[1].set_xlabel("Time (s)")
axs[0].set_ylabel("Signal")
axs[1].set_ylabel("Frequency (Hz)")
plt.suptitle("Sine signal")
plt.show()
```


    
![png](wavelets_files/wavelets_5_0.png)
    



    
![png](wavelets_files/wavelets_5_1.png)
    


## Perform a Continuous Wavelet Transform on the Signals

Using two implementations on two signals will give us four CWTs total.


```python
def cwt_pywt(time, y_signal, widths, wavelet="cmor1.5-1.0"):
    sampling_period = np.diff(time).mean()
    cwtmatr, freqs = pywt.cwt(y_signal, widths, wavelet, sampling_period=sampling_period)
    # take absolute value of complex result
    cwtmatr = np.abs(cwtmatr[:-1, :-1])
    return cwtmatr, 1/freqs
    

def cwt_scipy(time, y_signal, widths, w=6, wavelet=signal.morlet2):
    sampling_period = np.diff(time).mean()
    freqs = w / (2*np.pi*sampling_period*widths)
    cwtmatr = signal.cwt(y_signal, wavelet, widths)
    cwtmatr = np.abs(cwtmatr[:-1, :-1])#**2 / widths[:, np.newaxis]
    return cwtmatr, 1/freqs


# logarithmic scale for scales, as suggested by Torrence & Compo:
widths = np.geomspace(1, 1024, num=100)

chirp_power_pywt, chirp_period_pywt = cwt_pywt(time, chirp_signal, widths)
chirp_power_scipy, chirp_period_scipy = cwt_scipy(time, chirp_signal, widths)
sines_power_pywt, sines_period_pywt = cwt_pywt(time, sine_signal, widths)
sines_power_scipy, sines_period_scipy = cwt_scipy(time, sine_signal, widths)

# plot result using matplotlib's pcolormesh (image with annoted axes)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
kw = dict(shading="flat")

pcm1 = ax1.pcolormesh(time, chirp_period_pywt, chirp_power_pywt, **kw)
ylim1 = ax1.get_ylim()
ax1.plot(time, 1/frequency1, "w:")
ax1.plot(time, 1/frequency2, "w:")
ax1.set_yscale("log")
#ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Period (s)")
ax1.set_title("PyWavelets CWT of Chirps")
ax1.set_ylim(ylim1[::-1])
fig.colorbar(pcm1, ax=ax1)

pcm2 = ax2.pcolormesh(time, chirp_period_scipy, chirp_power_scipy, **kw)
ylim2 = ax2.get_ylim()
ax2.plot(time, 1/frequency1, "w:")
ax2.plot(time, 1/frequency2, "w:")
ax2.set_yscale("log")
#ax2.set_xlabel("Time (s)")
#ax2.set_ylabel("Period (s)")
ax2.set_title("SciPy CWT of Chirps")
ax2.set_ylim(ylim2[::-1])
fig.colorbar(pcm2, ax=ax2)

pcm3 = ax3.pcolormesh(time, sines_period_pywt, sines_power_pywt, **kw)
ylim3 = ax3.get_ylim()
ax3.hlines(periods, xmin=0, xmax=1, color="w", linestyle=":")
ax3.set_yscale("log")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Period (s)")
ax3.set_title("PyWavelets CWT of Sines")
ax3.set_ylim(ylim3[::-1])
fig.colorbar(pcm3, ax=ax3)

pcm4 = ax4.pcolormesh(time, sines_period_scipy, sines_power_scipy, **kw)
ylim4 = ax4.get_ylim()
ax4.hlines(periods, xmin=0, xmax=1, color="w", linestyle=":")
ax4.set_yscale("log")
ax4.set_xlabel("Time (s)")
#ax4.set_ylabel("Period (s)")
ax4.set_title("SciPy CWT of Sines")
ax4.invert_yaxis()
ax4.set_ylim(ylim4[::-1])
fig.colorbar(pcm4, ax=ax4)

fig.tight_layout()
```

    /tmp/ipykernel_196/190408064.py:12: DeprecationWarning: scipy.signal.cwt is deprecated in SciPy 1.12 and will be removed
    in SciPy 1.15. We recommend using PyWavelets instead.
    
      cwtmatr = signal.cwt(y_signal, wavelet, widths)



    
![png](wavelets_files/wavelets_7_1.png)
    


I don't understand why the chirp power spectra are offset from where the dominant frequencies should be. If it's offset for those, why is it not also offset for the sinusoid power spectra? The peaks in the sinusoidal power spectra are very slightly offset from the defined frequencies, but this is probably wavelet resolution effects or something.

## Everything below here is just playing around


```python
import matplotlib.pyplot as plt
import pywt
import numpy as np
wavelet = "cmor1.5-1.0"
fig, ax = plt.subplots(figsize=(4,4))
[psi, x] = pywt.ContinuousWavelet(wavelet).wavefun(10)
ax.plot(x, np.real(psi), label="real")
ax.plot(x, np.imag(psi), label="imag")
ax.set_title(wavelet)
ax.set_xlim([-5, 5])
ax.set_ylim([-0.8, 1])
```




    (-0.8, 1.0)




    
![png](wavelets_files/wavelets_10_1.png)
    



```python
periods = 2, 8, 32, 128, 512
t0, tf = 0, 13*365.25
t = np.arange(t0, tf, 0.1)
y = np.sum([np.sin(2*np.pi*t/p) for p in periods], axis=0)
plt.plot(t, y);
```


    
![png](wavelets_files/wavelets_11_0.png)
    



```python
# perform CWT
wavelet = "cmor1.5-1.0"
#wavelet = "cmor1.8-1.0"
#wavelet = "cmor0.5-2.0"
# logarithmic scale for scales, as suggested by Torrence & Compo:
freqs = np.geomspace(1/1024, 1, num=100)
widths = pywt.frequency2scale(wavelet, freqs/10)
sampling_period = np.diff(t).mean()
cwtmatr, freqs = pywt.cwt(y, widths, wavelet, sampling_period=sampling_period)
# absolute take absolute value of complex result
power1 = np.abs(cwtmatr[:-1, :-1])**2 / widths[:-1, np.newaxis] * np.sqrt(2*np.pi*1.5)

plt.close("all")
periods = 1/freqs
# plot result using matplotlib's pcolormesh (image with annoted axes)
fig, axs = plt.subplots()
pcm = axs.pcolormesh(t, periods, power1, vmax=1)
axs.set_yscale("log")
axs.set_xlabel("Time (s)")
axs.set_ylabel("Period (d)")
axs.set_title("Continuous Wavelet Transform (Scaleogram)")
axs.invert_yaxis()
fig.colorbar(pcm, ax=axs)

power = signal.cwt(y, signal.morlet2, widths)
power2 = np.abs(power[:-1, :-1])**2 / widths[:-1, np.newaxis]
fig, axs = plt.subplots()
pcm = axs.pcolormesh(t[:-1], periods[:-1], power2, vmax=1)
axs.set_yscale("log")
axs.set_xlabel("Time (s)")
axs.set_ylabel("Period (d)")
axs.set_title("Continuous Wavelet Transform (Scaleogram)")
axs.invert_yaxis()
fig.colorbar(pcm, ax=axs)
```

    /tmp/ipykernel_166/2780547923.py:25: DeprecationWarning: scipy.signal.cwt is deprecated in SciPy 1.12 and will be removed
    in SciPy 1.15. We recommend using PyWavelets instead.
    
      power = signal.cwt(y, signal.morlet2, widths)





    <matplotlib.colorbar.Colorbar at 0x7fceab1891d0>




    
![png](wavelets_files/wavelets_12_2.png)
    



```python
%pip install ssqueezepy
```

    Collecting ssqueezepy
      Downloading ssqueezepy-0.6.5-py3-none-any.whl.metadata (14 kB)
    Requirement already satisfied: numpy in /opt/conda/envs/notebook/lib/python3.11/site-packages (from ssqueezepy) (1.26.4)
    Requirement already satisfied: numba in /opt/conda/envs/notebook/lib/python3.11/site-packages (from ssqueezepy) (0.60.0)
    Requirement already satisfied: scipy in /opt/conda/envs/notebook/lib/python3.11/site-packages (from ssqueezepy) (1.14.1)
    Requirement already satisfied: llvmlite<0.44,>=0.43.0dev0 in /opt/conda/envs/notebook/lib/python3.11/site-packages (from numba->ssqueezepy) (0.43.0)
    Downloading ssqueezepy-0.6.5-py3-none-any.whl (127 kB)
    Installing collected packages: ssqueezepy
    Successfully installed ssqueezepy-0.6.5
    Note: you may need to restart the kernel to use updated packages.



```python
import ssqueezepy as sqpy
wx, scale = sqpy.cwt(y, wavelet="morlet", t=t)
p = 1/sqpy.experimental.scale_to_freq(scale, wavelet="morlet", N=len(t))

fig, axs = plt.subplots()
pcm = axs.pcolormesh(t, p, np.abs(wx)**2)
axs.set_yscale("log")
axs.set_xlabel("Time (s)")
axs.set_ylabel("Period (d)")
axs.set_title("Continuous Wavelet Transform (Scaleogram)")
axs.invert_yaxis()
fig.colorbar(pcm, ax=axs)
```




    <matplotlib.colorbar.Colorbar at 0x7fceaca47750>




    
![png](wavelets_files/wavelets_14_1.png)
    



```python
widths = np.geomspace(2, 4096, num=100)
time, period, power, phase = cwt(t, y, widths=widths)

plt.close("all")
fig, axs = plt.subplots()
pcm = axs.pcolormesh(t[:-1], period[:-1], power[:-1, :-1]**2 / widths[:-1, np.newaxis])
axs.set_yscale("log")
axs.set_xlabel("Time (s)")
axs.set_ylabel("Period (s)")
axs.set_title("Continuous Wavelet Transform (Scaleogram)")
axs.invert_yaxis()
fig.colorbar(pcm, ax=axs)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[21], line 2
          1 widths = np.geomspace(2, 4096, num=100)
    ----> 2 time, period, power, phase = cwt(t, y, widths=widths)
          4 plt.close("all")
          5 fig, axs = plt.subplots()


    NameError: name 'cwt' is not defined



```python

```
