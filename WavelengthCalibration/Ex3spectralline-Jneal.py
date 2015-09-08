#!/usr/bin/python
## Python for Astronomers
## Exercise 3: Time series of a spectral line
## Jason Neal 11 Decemeber 2014

## imports:
from astropy.modeling import models, fitting
import numpy as np 
import matplotlib.pyplot as plt
from astropy.modeling import SummedCompositeModel
from astropy.modeling.models import Gaussian1D
from astropy.modeling.models import custom_model_1d

## my functions:
def mk_3gaussians(amplitudes, means, stddevs, points=2000):
    '''Create 3 spectral lines by adding 3 gaussians together'''
    assert len(amplitudes) == len(means), ' Not the same length inputs'
    assert len(amplitudes) == len(stddevs), ' Not the same length inputs'
    x = np.linspace(300, 700, points)
    y = np.zeros(points)
    # g = models.Gaussian1D(amplitude=amplitudes, mean=means, stddev=stddevs)  # three gausians
    # eval_g = g(x,setmodel_set_axis=False)  # All 3 evaluated with x jsut need to sum the 3 together
    for i in range(len(amplitudes)):
        g = Gaussian1D(amplitude=amplitudes[i], mean=means[i], stddev=stddevs[i])
       # print g(x)
        y += g(x) # Adding each Gaussian  
    y +=  np.random.normal(0, 3, x.shape)		# Adding some noise to our data
    return x, y

# def Gaussians3_fiter(amp1, amp2, amp3, mean1, mean2, mean3, std1, std2, std3):
#     ''' Combined 3 gausian model made using sum combined model to use for fitting
# Note: Composite models in Astropy are currently in the process of being reworked, 
# but in the meantime the existing implementation is still useful.
# Are not able to fit composite models apparently!!'''
#     g1 = Gaussian1D(amplitude=amp1, stddev=std1, mean=mean1)
#     g2 = Gaussian1D(amplitude=amp2, stddev=std2, mean=mean2)
#     g3 = Gaussian1D(amplitude=amp3, stddev=std3, mean=mean3)
#     summed_3_g = SummedCompositeModel([g1, g2, g3])
#     return summed_3_g 

@custom_model_1d
def sum_of_gaussians(x, amplitude1=120., mean1=351., sigma1=15.,
                        amplitude2=100., mean2=505., sigma2=15.,
                        amplitude3=100., mean3=595., sigma3=15):
    ''' Could also use this to generate the data.
    Has been adjusted from http://docs.astropy.org/en/stable/modeling/new.html'''
    return (amplitude1 * np.exp(-0.5 * ((x - mean1) / sigma1)**2) +
            amplitude2 * np.exp(-0.5 * ((x - mean2) / sigma2)**2) + 
    amplitude3 * np.exp(-0.5 * ((x - mean3) / sigma3)**2))


def rv_shift(x, RV, period, phase=0, number=10):
    '''RV shift spectrum a number of times.
    This does it by shifting the wavelenght coordinate by a sine function'''
    shifts = np.zeros([number,len(x)])
    t = np.array(range(number))
    #doppler = RV * np.sin(np.pi * t / period + phase)
    d = doppler_rv(amp=RV, period=period,  phase=phase)
    doppler = d(t)
    print(doppler)
    for i in range(number):
    	shifts[i, :] = x + doppler[i]
    return shifts

@custom_model_1d
def doppler_rv(t, amp=10, period=4, phase=.2):
    ''' custom Sine function '''
    return amp * np.sin(np.pi * t / period + phase)


# Main program:
def main():
    ''' Very NON-pythonic method of solving''' 
    num_shifts = 15    # number of rv shifts to do
    # parameters for the gausians
    amplitudes = [100, 50, 75] 
    means = [350, 500, 600]
    stddevs = [5, 10, 7]
    # could use the fitting function to make instead of this
    x, y = mk_3gaussians(amplitudes, means, stddevs, 1000)
    plt.plot(x, y)
    plt.xlabel('Wavelength')
    plt.show()
    

    # Change x positions to RV shift data
    # simulate RV shift by just changing wavelength
    RV_val = 11.4
    period_val = 3.7
    phase_value = 45 * np.pi / 180
    rvshifted = rv_shift(x, RV=RV_val, period=period_val, phase=phase_value, number=num_shifts)
    for i in range(10):
        plt.plot(rvshifted[i, :], y - 20 * i, lw=3)
    plt.xlabel('Wavelength', fontsize=14)
    plt.title('RV shifted lines', fontsize=18)
    plt.show()   # Ugly plot

    # Fit Summed Gausian
    g_init = sum_of_gaussians()
    fit_g = fitting.LevMarLSQFitter()
    sol_g = fit_g(g_init, x, y)
    plt.plot(x, y, label='data')
    plt.plot(x, g_init(x), label='guess')
    plt.plot(x, sol_g(x), label='fit')
    plt.legend()
    plt.show()

    # Find shifted mean of each gausian peak for all rv shifts
    means1 = np.zeros(num_shifts)
    means2 = np.zeros(num_shifts)
    means3 = np.zeros(num_shifts)
    for i in range(num_shifts):
        if i == 0:
            solution = fit_g(g_init, rvshifted[i, :], y)
        else:
            solution = fit_g(past_sol, rvshifted[i, :], y)
        means1[i] = solution.mean1.value
        means2[i] = solution.mean2.value
        means3[i] = solution.mean3.value #(solution.mean1, solution.mean2, solution.mean3)
        past_sol = solution
        # plt.plot(rvshifted[i, :], y, label='data')
        # plt.plot(rvshifted[i, :], solution(rvshifted[i, :]), label='Fit')
        # plt.legend()
        # plt.show()
    means1 -= np.median(means1)  # subtract median wavelenght value
    means2 -= np.median(means2) 
    means3 -= np.median(means3)

 # fit sine to each means data
    t = np.array(range(num_shifts))
    # s_init = models.Sine1D(amplitude=25,frequency=1/3)
    s_init = doppler_rv()
    fit_s = fitting.LevMarLSQFitter()
    sol_s_mean1 = fit_s(s_init, t, means1)
    sol_s_mean2 = fit_s(s_init, t, means2)
    sol_s_mean3 = fit_s(s_init, t, means3)
    print(sol_s_mean1)
    print(sol_s_mean2)
    print(sol_s_mean3)

    fit_t = np.linspace(0,num_shifts,300)  # points to plot
    found_RV = round(np.median([sol_s_mean1.amp.value, sol_s_mean2.amp.value, sol_s_mean3.amp.value]), 3)
    found_period = round(np.median([sol_s_mean1.period.value, sol_s_mean2.period.value, sol_s_mean3.period.value]), 3)
    found_phase = round(np.median([sol_s_mean1.phase.value, sol_s_mean2.phase.value, sol_s_mean3.phase.value]), 3)
   
    plt.plot(t, means1, 'r*', label='1st line')
    plt.plot(t, means2+1, 'b*', label='2nd line')
    plt.plot(t, means3+2, 'g*', label='3rd line')
    plt.plot(fit_t, sol_s_mean1(fit_t), 'r', label='fit1')
    plt.plot(fit_t, sol_s_mean2(fit_t)+1, 'b', label='fit2')
    plt.plot(fit_t, sol_s_mean3(fit_t)+2, 'g', label='fit3')
    plt.title('Fittting Lineshifts')
    plt.ylabel('Wavelength shift')
    plt.xlabel('Even measurements in time')
    plt.text(0.2, -13, 'Given values RV = {},     period = {},     phase = {}'.format(RV_val, period_val, phase_value))
    plt.text(0.1, -14, 'Found Values RV = {}, period = {}, phase = {}'.format(found_RV, found_period, found_phase))
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
