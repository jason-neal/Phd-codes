import mpmath as mp
import numpy as np
from mpmath import mpf


def BIC_fixed_snr_chi2(chi_square, n_data, n_params, sigma_fixed):
    a = (1 / mp.sqrt(2 * mp.pi)) ** n_data

    c = mp.exp(-chi_square / 2)
    likelyhood = ((1 / mpf(sigma_fixed)) ** n_data) * a * c

    # log_likelyhood = n_data*(np.log(1/np.sqrt(2*np.pi)) + 1/sigma_fixed) - chi_square / 2

    bic = (n_params * mp.log(n_data)) - 2 * mp.log(likelyhood)
    return bic


def BIC_fixed_snr_chi2old(chi_square, n_data, n_params, sigma_fixed):
    # a = (1/np.sqrt(2 * np.pi)) ** n_data
    # b = ((1/sigma_fixed) ** n_data)
    # c = np.exp(-chi_square / 2)
    # likelyhood = a * b * c

    log_likelyhood = n_data * (np.log(1 / np.sqrt(2 * np.pi)) + 1 / sigma_fixed) - chi_square / 2

    bic = n_params * np.log(n_data) - 2 * log_likelyhood
    return bic


def BIC_obs_snr_chi2(chi_square, n_data, n_params, sigma_vect):
    log_likelyhood = n_data * (np.log(1 / np.sqrt(2 * np.pi))) + np.log(
        np.prod(sigma_vect)) - chi_square / 2

    bic = n_params * np.log(n_data) - 2 * log_likelyhood
    return bic


if __name__ == "__main__":
    chi1 = 4978
    chi2 = 2792
    bic1 = BIC_fixed_snr_chi2(chi1, 3072, 2, 1 / 150)
    bic2 = BIC_fixed_snr_chi2(chi2, 3072, 4, 1 / 150)
    print("BIC1", bic1)
    print("BIC2", bic2)
    print("Delta 2- 1", bic2 - bic1)

    chi3 = 3746
    chi4 = 3630
    bic3 = BIC_fixed_snr_chi2(chi3, 3072, 2, 1 / 150)
    bic4 = BIC_fixed_snr_chi2(chi4, 3072, 4, 1 / 150)
    print("BIC3", bic4)
    print("BIC4", bic3)
    print("Delta 4-3", bic4 - bic3)

    chi5 = 37688
    pixel_chi5 = 2612
    snr_chi5 = 272
    chi6 = 33860
    pixel_chi6 = 2612
    snr_chi6 = 272
    bic5 = BIC_fixed_snr_chi2(chi5, pixel_chi5, 2, 1 / 150)
    bic6 = BIC_fixed_snr_chi2(chi6, pixel_chi6, 4, 1 / 150)

    print("est BIC5 actual ", BIC_fixed_snr_chi2(chi5, 2072, 2, 1 / snr_chi5))
    print("est BIC6 actual ", BIC_fixed_snr_chi2(chi6, 2072, 4, 1 / snr_chi6))
    print("Delta BIC6-bic5 = ", bic6 - bic5)

    print("est BIC5 2000 300", BIC_fixed_snr_chi2(chi5, 2072, 2, 1 / 300))
    print("est BIC6 2000 300", BIC_fixed_snr_chi2(chi6, 2072, 4, 1 / 300))

    print("est BIC5 1000 150 ", BIC_fixed_snr_chi2(chi5, 1072, 2, 1 / 150))

    print("est BIC6 1000 150 ", BIC_fixed_snr_chi2(chi6, 1072, 4, 1 / 150))

    print("est BIC5 3000 150 ", BIC_fixed_snr_chi2(chi5, 3072, 2, 1 / 150))

    print("est BIC6 3000 150 ", BIC_fixed_snr_chi2(chi6, 3072, 4, 1 / 150))

    print("delta BIC5 2000 150 ",
          BIC_fixed_snr_chi2(chi5, 2072, 2, 1 / 150) - BIC_fixed_snr_chi2(chi6, 2072, 4, 1 / 150))

    print("delta 2000 300",
          BIC_fixed_snr_chi2(chi5, 2072, 2, 1 / 300) - BIC_fixed_snr_chi2(chi6, 2072, 4, 1 / 300))

    print("delta 1000 150 ",
          BIC_fixed_snr_chi2(chi5, 1072, 2, 1 / 150) - BIC_fixed_snr_chi2(chi6, 1072, 4, 1 / 150))

    print("delta 3000 150 ",
          BIC_fixed_snr_chi2(chi5, 3072, 2, 1 / 150) - BIC_fixed_snr_chi2(chi6, 3072, 4, 1 / 150))
