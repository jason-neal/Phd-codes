#!/usr/bin/env python
# Plot the chisquare for alpha detection
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import numba
import pickle

def main():
    """ Plot the chi squared"""
    path = "/home/jneal/Phd/Codes/Phd-codes/Simulations/saves"
    X = np.load(os.path.join(path, "RV_mesgrid.npy"))
    Y = np.load(os.path.join(path, "alpha_meshgrid.npy"))
    snrs = np.load(os.path.join(path, "snr_values.npy"))
    Resolutions = np.load(os.path.join(path, "Resolutions.npy"))
    # chisqr_store = np.load(os.path.join(path, "chisquare_data.npy"))
# scipy_chisqr_store = np.load(os.path.join(path, "scipy_chisquare_data.npy"))
    # X, Y = np.meshgrid(RVs, alphas)
    chisqr_snr = dict()
    for snr in snrs:
        chisqr_snr[snr] = np.load(os.path.join(path,
                                  "scipy_chisquare_data_snr_{}.npy".format(snr)
    # unpickle parameter
    try:
        with open(os.path.join(path, "input_params.pickle"),"rb") as f:
            input_parameters = pickle.load(f)
    except:
        raise
        input_parameters = (999, 999)
                                               ))

    df_list = []   # To make data frame
    for resolution in Resolutions:
        for snr in snrs:
            this_chisqr_snr = res_chisqr_snr[resolution][snr]
            this_error_chisqr_snr = res_error_chisqr_snr[resolution][snr]
            # Calculating the minimum location
            print("snr = ", snr)
            #print("min chisquared value", np.min(this_chisqr_snr),
            #      "location", np.argmin(this_chisqr_snr))
            #print("min scipy chisquared value", np.min(this_chisqr_snr),
            #      "location", np.argmin(this_chisqr_snr))
            print("\n Using expectation value chi sqaure")
            print("min scipy chisquared value",
                  np.min(this_chisqr_snr, axis=(0, 1)), "location",
                  np.argmin(this_chisqr_snr))
            print("RV = ", X.ravel()[np.argmin(this_chisqr_snr)], "APLHA = ", Y.ravel()[np.argmin(this_chisqr_snr)])
            print("\n Using sigma value chi sqaure")
            print("min scipy chisquared value",
                  np.min(this_error_chisqr_snr, axis=(0, 1)), "location",
                  np.argmin(this_error_chisqr_snr))
            print("RV = ", X.ravel()[np.argmin(this_error_chisqr_snr)], "APLHA = ", Y.ravel()[np.argmin(this_error_chisqr_snr)], "\n")

            rv_at_min = X.ravel()[np.argmin(this_chisqr_snr)]
            alpha_at_min = Y.ravel()[np.argmin(this_chisqr_snr)]
            df_list.append([resolution, snr, rv_at_min, alpha_at_min, np.min(this_error_chisqr_snr, axis=(0, 1))])
    df = pd.DataFrame(df_list, columns=["Resolution", "SNR", "Recovered RV",
                                        "Recovered Alpha", "chi**2"])
    print(df)

    # plt.contourf(X, Y, np.log(chis
    # plt.contourf(X, Y, np.log(chisqr_store), 40)
    # plt.title("my chisquared ".format(snr))
    # plt.show()
    for snr in snrs:
        this_chisqr_snr = chisqr_snr[snr]
        plt.contourf(X, Y, np.log(this_chisqr_snr), 100)
        plt.title("Log Chi squared with snr of {}".format(snr))
        plt.show()
        print("snr = ", snr)
        print("min chisquared value", np.min(this_chisqr_snr),
              "location", np.argmin(this_chisqr_snr))
        print("min scipy chisquared value", np.min(this_chisqr_snr),
              "location", np.argmin(this_chisqr_snr))

    for snr in snrs:
        this_chisqr_snr = chisqr_snr[snr]
        plt.contourf(X, Y, this_chisqr_snr, 100)
        plt.title("Chi squared with snr of {}".format(snr))
        plt.show()


if __name__ == "__main__":
    main()
