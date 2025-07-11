import numpy as np
from scipy.stats import binom
from scipy.signal import convolve
from scipy.integrate import cumtrapz
from tulap import random as tulap_random
import pandas as pd
import matplotlib.pyplot as plt

# def tulap_pmf(epsilon=0.1):
#     de=0
#     b = np.exp(-epsilon)
#     q = 2 * de * b / (1 - b + 2 * de * b)
#     tulap_sample = tulap_random(n=100000, m=0, b=b, q=q)

#     """
#     Estimate the PMF of a Tulap sample by binning to integer values from 0 to num_samples.
#     Returns a numpy array of probabilities for each integer value.
#     """
#     #this isn't what we want. how to add tulap to binomial? can't convolve b/c tulap is continuous
#     #maybe for each num_samples, run a ton of trials sampling from binomial and tulap and adding noise
#     #yeah
#     bins = np.arange(0, num_samples + 2)  # +2 so last bin edge is inclusive for num_samples
#     hist, _ = np.histogram(tulap_sample, bins=bins)
#     tulap_pmf = hist / np.sum(hist)

#     print(tulap_pmf)
#     return tulap_pmf

# def tulap_binomial_cdf_ppf(num_samples:int, tulap_pmf, p=0.3):

#     n_vals = range(num_samples)

#     noiseless = [binom.pmf(k, num_samples, p) for k in n_vals]

#     print(noiseless)

#     pmf_noisy = convolve(tulap_pmf,noiseless)

#     # Calculate the CDF by cumulatively integrating the PDF of the sum
#     # Assuming a uniform step size for integration (e.g., 1 if the bins are unit width)
#     cdf_noisy = cumtrapz(pmf_noisy, initial=0)

#     noisy_ppf = np.searchsorted(cdf_noisy, 0.05)

#     return cdf_noisy, noisy_ppf

def empirical_noisy_binomial(num_samples:int, epsilon=0.1, p=0.3):
    num_trials = 100000
    de=0
    b = np.exp(-epsilon)
    q = 2 * de * b / (1 - b + 2 * de * b)
    tulap_sample = tulap_random(n=num_trials, m=0, b=b, q=q)
    
    # Generate binomial sample
    binomial_sample = binom.rvs(n=num_samples, p=p, size=num_trials)
    
    # Add Tulap noise to the binomial sample
    noisy_binomial_sample = binomial_sample + tulap_sample

    # Compute the empirical CDF
    sorted_sample = np.sort(noisy_binomial_sample)
    cdf_vals = np.arange(1, len(sorted_sample)+1) / len(sorted_sample)

    idx = np.searchsorted(cdf_vals, 0.05)
    k_crit = sorted_sample[idx] if idx < len(sorted_sample) else sorted_sample[-1]

    return cdf_vals, k_crit, noisy_binomial_sample

def compute_power_df(metadata_df, combo, alt_probs=[0.5,0.6], campaign_size=1000):
    
    print(metadata_df)
    results = []


    for num_samples in range(1, campaign_size+1):
        for alt_prob in alt_probs:
            for alpha_targeting, alpha_engagement, epsilon in combo:
                p_alt = metadata_df.loc[(alt_prob, alpha_targeting, alpha_engagement), "conversionProbAdA"]
                p_null = metadata_df.loc[(alt_prob, alpha_targeting, alpha_engagement), "conversionProbAdA_null"]
                epsilon = epsilon[0]
                if epsilon > 0:
                    #print(f"Calculating power for alt_prob={p_alt}, null_prob={p_null}, epsilon={epsilon}, num_samples={num_samples}")
                    cdf_null, k_crit_null, null_binom = empirical_noisy_binomial(num_samples=num_samples, epsilon=epsilon, p=p_null)
                    cdf_alt, k_crit_alt, alt_binom = empirical_noisy_binomial(num_samples=num_samples, epsilon=epsilon, p=p_alt)
                    density_above_k_crit = np.mean(alt_binom > k_crit_null)
                    #print(f"Density (probability) of alt_binom above k_crit_null: {density_above_k_crit:.4f}")
                    power = 1 - density_above_k_crit
                    results.append({'alt_prob': p_alt, 'null_prob':p_null, 'epsilon' : epsilon, 'num_samples': num_samples, 'power': power})

                    if num_samples%1000 == 0:
                        # Plot PDF and mark k_crit
                        plt.figure(figsize=(8, 5))
                        counts, bin_edges, _ = plt.hist(null_binom, bins=50, density=True, alpha=0.6, label="Empirical PDF Null")
                        plt.axvline(x=k_crit_null, color='blue', linestyle='--', label=f'k_crit={k_crit_null:.2f}')
                        counts, bin_edges, _ = plt.hist(alt_binom, bins=50, density=True, alpha=0.6, color='red', label="Empirical PDF Alt")
                        plt.axvline(x=k_crit_alt, color='red', linestyle='--', label=f'k_crit={k_crit_alt:.2f}')
                        plt.xlabel("Value")
                        plt.ylabel("Density")
                        plt.title("Empirical PDF of Noisy Binomial Sample")
                        plt.legend()
                        plt.tight_layout()
                        plt.show()
    
    # Convert to DataFrame with alt_prob and epsilon as index and num_samples as columns
    df = pd.DataFrame(results)
    df_pivot = df.pivot_table(index=['alt_prob', 'null_prob', 'epsilon'], columns='num_samples', values='power')
    print(df_pivot)
    return df_pivot

#save power df to folder named by plot type. do not clean this as part of clean!! save it by alt_prob so can be used by parallel
#(and so it's smaller and the different alt probs can stop early if they want)
#i.e. rewrite so it stops when power >= 0.9


# # print("cdf_null:", cdf_null, "k_crit_null:", k_crit_null)
# # print("cdf_alt:", cdf_alt, "k_crit_alt:", k_crit_alt)

# # print("length cdf_null:", len(cdf_null), "length cdf_alt:", len(cdf_alt))

# # # Plot CDF and k_crit for null
# # plt.figure(figsize=(8, 5))
# # plt.plot(np.sort(np.arange(len(cdf_alt))), cdf_alt, label="Empirical CDF (null)")
# # plt.axvline(x=k_crit_null, color='red', linestyle='--', label=f'k_crit_null={k_crit_null:.2f}')
# # plt.axvline(x=k_crit_null*1000, color='red', linestyle='--', label=f'k_crit_null={k_crit_null*1000:.2f}')
# # plt.xlabel("Sorted Sample Index")
# # plt.ylabel("Empirical CDF")
# # plt.title("Empirical CDF and Critical Value (Null)")
# # plt.legend()
# # plt.tight_layout()
# # plt.show()


# # Plot PDF and mark k_crit
# plt.figure(figsize=(8, 5))
# counts, bin_edges, _ = plt.hist(null_binom, bins=50, density=True, alpha=0.6, label="Empirical PDF Null")
# plt.axvline(x=k_crit_null, color='blue', linestyle='--', label=f'k_crit={k_crit_null:.2f}')
# counts, bin_edges, _ = plt.hist(alt_binom, bins=50, density=True, alpha=0.6, color='red', label="Empirical PDF Alt")
# plt.axvline(x=k_crit_alt, color='red', linestyle='--', label=f'k_crit={k_crit_alt:.2f}')
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.title("Empirical PDF of Noisy Binomial Sample")
# plt.legend()
# plt.tight_layout()
# plt.show()

# #this is not working idk.
# # power = cdf_alt[int(k_crit_null*1000)]

# # print(f"Type II error (power = 1 - type II error): {power}")

# num_samples = 10
# p_null = 0.3
# p_alt = 0.2
# epsilon = 0.1

# tulap_pmf = tulap_pmf(epsilon=epsilon)

# cdf_noisy_null, k_crit_noisy_null = tulap_binomial_cdf_ppf(num_samples, tulap_pmf, p=p_null)

# # Suppose you want to store type_2_error for each alt_prob and num_samples
# results = []

# for alt_prob in [0.1, 0.2, 0.3]:  # example alt_probs
#     for num_samples in [100]:  # example num_samples
#         cdf_noisy_alt, _ = tulap_binomial_cdf_ppf(num_samples, tulap_pmf, p=alt_prob)
#         type_2_error = cdf_noisy_alt[k_crit_noisy_null]
#         results.append({'alt_prob': alt_prob, 'num_samples': num_samples, 'type_2_error': type_2_error})

# # Convert to DataFrame with alt_prob as index and num_samples as columns
# df = pd.DataFrame(results)
# df_pivot = df.pivot(index='alt_prob', columns='num_samples', values='type_2_error')
# print(df_pivot)