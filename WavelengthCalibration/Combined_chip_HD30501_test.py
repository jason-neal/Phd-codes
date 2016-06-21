#!/usr/lib/python3

# Testing the affectof using all 3 cips with the known offset between them to wave length calibrate the spectra
import numpy as np
import matplotlib.pyplot as plt

def gen_map(x, a, b, c, noise):
    #noise_vals = noise * np.random.randn(len(x))
    ans =  [a*xpos**2 + b*xpos + c + float(noise*np.random.randn(1)) for xpos in x]
    return ans

# Pixel gaps Brogi et al 2015
Gap1 = 282
Gap2 = 278
Gap3 = 275
Gap_sum = Gap1 + Gap2 + Gap3

Chipnames = ["Coordinates_CRIRE.2012-04-07T00:08:29.976_1.nod.ms.norm.sum.txt","Coordinates_CRIRE.2012-04-07T00:08:29.976_2.nod.ms.norm.sum.txt","Coordinates_CRIRE.2012-04-07T00:08:29.976_3.nod.ms.norm.sum.txt","Coordinates_CRIRE.2012-04-07T00:08:29.976_4.nod.ms.norm.sum.txt"]
PATH = "/home/jneal/Dropbox/PhD/"
#PATH = "/home/jneal/Dropbox/PhD/"
#"/home/jneal/Dropbox/PhD/"
pix1, wlen1, dpth1 = np.loadtxt(PATH+Chipnames[0], skiprows=1, unpack=True)
pix2, wlen2, dpth2 = np.loadtxt(PATH+Chipnames[1], skiprows=1, unpack=True)
pix3, wlen3, dpth3 = np.loadtxt(PATH+Chipnames[2], skiprows=1, unpack=True)
pix4, wlen4, dpth4 = np.loadtxt(PATH+Chipnames[3], skiprows=1, unpack=True)
while False:
	pass
	#Test_pxl1 = [70, 200, 549, 937, 1015]
	#Test_pxl2 = [100, 400, 649, 737, 815] 
	#Test_pxl3 = [50, 200, 549, 937, 915] 
	#Test_pxl4 = [207, 400, 519, 837, 1015] 

	#Test_pxl2 = [pxl + 1*1024 + Gap1 for pxl in Test_pxl2]
	#Test_pxl3 = [pxl + 2*1024 + Gap1 + Gap2 for pxl in Test_pxl3]
	#Test_pxl4 = [pxl + 3*1024 + Gap_sum for pxl in Test_pxl4]

	#aa = 5/5000000.0    # smaller value 
	#bb = 80/5000.0   # 80 nm over 4*1024 detectors plus gaps
	## noise = 0.05   # nm

	##Test_wl2 = gen_map(Test_pxl2, aa, bb, cc, noise)
	#Test_wl3 = gen_map(Test_pxl3, aa, bb, cc, noise)
	#Test_wl4 = gen_map(Test_pxl4, aa, bb, cc, noise)
 
Test_pxl1 = [pxl for pxl in pix1] 
Test_pxl2 = [pxl + 1*1024 + Gap1 for pxl in pix2]
Test_pxl3 = [pxl + 2*1024 + Gap1 + Gap2 for pxl in pix3]
Test_pxl4 = [pxl + 3*1024 + Gap_sum for pxl in pix4]
Test_wl1 = [wl for wl in wlen1]
Test_wl2 = [wl for wl in wlen2]
Test_wl3 = [wl for wl in wlen3]
Test_wl4 = [wl for wl in wlen4]

plt.subplot(211)
plt.plot(Test_pxl1, Test_wl1, "bo", label="1")
plt.plot(Test_pxl2, Test_wl2, "ro", label="2")
plt.plot(Test_pxl3, Test_wl3, "go", label="3")
plt.plot(Test_pxl4, Test_wl4, "mo", label="4")
plt.ylim([min(Test_wl1), max(Test_wl4)])
ax1 = plt.gca()
ax1.get_yaxis().get_major_formatter().set_useOffset(False)
plt.xlabel("Pixel Position")
plt.ylabel("Wavelength")
#plt.legend()
#plt.show()

max_pixel = 4*1024 + Gap_sum
pixel_span = range(1, max_pixel)

# Fit to the individual chips
order = 2
wl_map1 = np.polyfit(Test_pxl1, Test_wl1, order)
print("\nwl_map params 1\t", wl_map1)
wlvals1 = np.polyval(wl_map1, pixel_span) 

wl_map2 = np.polyfit(Test_pxl2, Test_wl2, order)
print("\nwl_map params 2\t", wl_map2)
wlvals2 = np.polyval(wl_map2, pixel_span) 

wl_map3 = np.polyfit(Test_pxl3, Test_wl3, order)
print("\nwl_map params 3 \t", wl_map3)
wlvals3 = np.polyval(wl_map3, pixel_span) 

wl_map4 = np.polyfit(Test_pxl4, Test_wl4, order)
print("\nWl map params Combined\t", wl_map4)
wlvals4 = np.polyval(wl_map4, pixel_span) 

# Fit to combined data
Combined_pixels = Test_pxl1 + Test_pxl2 + Test_pxl3 + Test_pxl4
Combined_wls = Test_wl1 + Test_wl2 + Test_wl3 + Test_wl4

print("len(Combined_pixels)",len(Combined_pixels))

print("len(Combined_wls)",len(Combined_wls))
print(Combined_wls)

Combined_map = np.polyfit(Combined_pixels, Combined_wls, order)
print("\nwl_map params 4\t", Combined_map)
Combined_vals = np.polyval(Combined_map, pixel_span) 


plt.plot(pixel_span, wlvals1, 'b', label="Chip1")
plt.plot(pixel_span, wlvals2, 'r', label="Chip2")
plt.plot(pixel_span, wlvals3, 'g', label="Chip3")
plt.plot(pixel_span, wlvals4, 'm', label="Chip4")
plt.plot(pixel_span, Combined_vals, 'k', label="Combined")

plt.legend(loc=0)

plt.subplot(212)
plt.plot(pixel_span, np.zeros_like(pixel_span), "k--", label = "")
plt.plot(pixel_span, Combined_vals-wlvals1, 'b', label = "Combined-Chip1")
plt.plot(pixel_span, Combined_vals-wlvals2, 'r', label = "Combined-Chip2")
plt.plot(pixel_span, Combined_vals-wlvals3, 'g', label = "Combined-Chip3")
plt.plot(pixel_span, Combined_vals-wlvals4, 'm', label = "Combined-Chip4")
plt.xlabel("Pixel Position")
plt.ylabel("Wavelength Difference\nbetween Models (nm)")
plt.legend(loc=0)
ax1 = plt.gca()
ax1.get_yaxis().get_major_formatter().set_useOffset(False)
plt.ylim([-0.3,0.3])

# Mark Chip positions
linepos = [0, 1024, 1024+Gap1, 2*1024+Gap1, 2*1024+Gap1+Gap2, 3*1024+Gap1+Gap2, 3*1024+Gap1+Gap2+Gap3, 4*1024+Gap1+Gap2+Gap3]
plt.vlines(linepos, -1, 1)

plt.show()




#  Try fitting with matrix multiplication

x = np.array(Combined_pixels)
y = np.array(Combined_wls)
yerr = np.array(noise*np.ones_like(x))
#print("x", x)
#print("y", y)

## least-squares solution
# build the matrix A (you can use the np.vstack function)
A = np.vstack((np.ones_like(x), x, x**2)).T

# build the matrix C
C = np.diag(yerr**2)

# calculate the covariance matrix [A^T C^-1 A]^-1
# (use the linear algebra functions in np.linalg)
cov = np.dot(A.T, np.linalg.solve(C, A))
cov2 = np.linalg.inv(cov)

# calculate the X matrix
X = np.dot(cov2, np.dot(A.T, np.linalg.solve(C, y)))

# extract from X the parameters m and b
b, m, q = X 

print('b= {} +- {}'.format(b, np.sqrt(cov2[0,0])))
print('m= {} +- {}'.format(m, np.sqrt(cov2[1,1])))
print('q= {} +- {}'.format(q, np.sqrt(cov2[2,2])))

# plot the data (with errorbars) and the best-fit line
plt.figure()
plt.errorbar(x, y, yerr=noise, fmt='o')

xx = np.linspace(min(x), max(x))
plt.plot(xx, q*xx**2 + m*xx + b, '-', label="Matrix LR")
plt.title("Matrix multiplication")
plt.plot(pixel_span, Combined_vals, 'k', label="PolyVal Combined")
plt.legend()
plt.show()


print("They return the equivalent results")


 #Third order regression
print("adding x**3 term ")
A = np.vstack((np.ones_like(x), x, x**2, x**3)).T

# build the matrix C
C = np.diag(yerr**2)

# calculate the covariance matrix [A^T C^-1 A]^-1
# (use the linear algebra functions in np.linalg)
cov = np.dot(A.T, np.linalg.solve(C, A))
cov2 = np.linalg.inv(cov)

# calculate the X matrix
X = np.dot(cov2, np.dot(A.T, np.linalg.solve(C, y)))

# extract from X the parameters m and b
b3, m3, q3 , r3 = X 

print('b= {} +- {}'.format(b3, np.sqrt(cov2[0,0])))
print('m= {} +- {}'.format(m3, np.sqrt(cov2[1,1])))
print('q= {} +- {}'.format(q3, np.sqrt(cov2[2,2])))
print('r= {} +- {}'.format(r3, np.sqrt(cov2[3,3])))

# plot the data (with errorbars) and the best-fit line
plt.figure()
plt.subplot(211)
plt.errorbar(x, y, yerr=noise, fmt='o')

#xx = np.linspace(min(x), max(x))
xx = np.array(pixel_span)
yy = r3*xx**3 + q3*xx**2 + m3*xx + b3
plt.plot(xx, yy, '-', label="Matrix LR")
plt.title("Matrix multiplication 3rd order ")
plt.plot(pixel_span, Combined_vals, 'k', label="PolyVal Combined")
plt.legend()

plt.subplot(212)
plt.plot(pixel_span, yy-Combined_vals)
plt.title("Difference between x**3 term and x**2 fits")
plt.show()

print("Orginal Equation = {0}*x**2 + {1}*x + {2}".format(aa, bb, cc))

print("Polyval Equation = {0}*x**2 + {1}*x + {2}".format(Combined_map[0], Combined_map[1], Combined_map[2]))

print("Regression Equation = {0}*x**2 + {1}*x + {2}".format(q, m, b))

print("x**3 Regression Equation = {0}*x**3 + {1}*x**2 + {2}*x + {3}".format(r3, q3, m3, b3))