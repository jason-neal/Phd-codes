#!/usr/lib/python3

# Testing the affectof using all 3 cips with the known offset between them to wave length calibrate the spectra
import numpy as np
import matplotlib.pyplot as plt

def gen_map(x, a, b, c, noise):
    #noise_vals = noise * np.random.randn(len(x))
    ans =  [a*xpos**2 + b*xpos + c + noise*np.random.randn(1) for xpos in x]
    return ans

#Pixel gaps Brogi et al 2015
Gap1 = 282
Gap2 = 278
Gap3 = 275

Test_pxl1 = [70, 200, 549, 937, 1015]
Test_pxl2 = [100, 400, 649, 737, 815] 
Test_pxl3 = [50, 200, 549, 937, 915] 
Test_pxl4 = [207, 400, 519, 837, 1015] 

Test_pxl2 = [pxl + 1*1024 + Gap1 for pxl in Test_pxl2]
Test_pxl3 = [pxl + 2*1024 + Gap1 + Gap2 for pxl in Test_pxl3]
Test_pxl4 = [pxl + 3*1024 + Gap1 + Gap2 + Gap3 for pxl in Test_pxl4]

a = 5/500.0    # smaller value 
b = 800/5000.0   # 80 nm over 4*1024 detectors plus gaps
c = 2110.0      # nm
noise = 0.005  # nm

Test_wl1 = gen_map(Test_pxl1, a, b, c, noise)
Test_wl2 = gen_map(Test_pxl2, a, b, c, noise)
Test_wl3 = gen_map(Test_pxl3, a, b, c, noise)
Test_wl4 = gen_map(Test_pxl4, a, b, c, noise)
 

plt. plot(Test_pxl1, Test_wl1, "*",label="1")
plt. plot(Test_pxl2, Test_wl2, "*", label="2")
plt. plot(Test_pxl3, Test_wl3, "*", label="3")
plt. plot(Test_pxl4, Test_wl4, "*", label="4")
plt.ylim([min(Test_wl1),max(Test_wl4)])
ax1 = plt.gca()
ax1.get_yaxis().get_major_formatter().set_useOffset(False)

plt.legend()
plt.show()


# Fit to the individual chips




# Fit to combined data



