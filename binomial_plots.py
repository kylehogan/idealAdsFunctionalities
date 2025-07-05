import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

def plot_binomial_power_curve_subplots(alt_probs, alpha=0.05, max_samples=500, step=1, metadata_df=None, alpha_targeting=0.5, alpha_engagement=0.5):
    """
    Plot power vs. number of samples for a binomial test comparing each alt_prob to null_prob as subplots.
    """
    num_probs = len(alt_probs)
    fig, axes = plt.subplots(1, num_probs, figsize=(6 * num_probs, 5), sharey=True)
    if num_probs == 1:
        axes = [axes]
    ns = np.arange(1, max_samples + 1, step)
    for i, alt_prob in enumerate(alt_probs):
        p_null = metadata_df.loc[(alt_prob, alpha_targeting, alpha_engagement), "conversionProbAdA_null"]
        p_alt = metadata_df.loc[(alt_prob, alpha_targeting, alpha_engagement), "conversionProbAdA"]
        powers = []
        for n in ns:
            # One-sided less test
            k_crit = binom.ppf(alpha, n, p_null)
            power = binom.cdf(k_crit, n, p_alt)
            print(f"alt_prob={alt_prob}, n={n}, k_crit={k_crit}, power={power:.3f}")
            powers.append(power)
        ax = axes[i]
        ax.plot(ns, powers, label=f'Power (alt={p_alt}, null={p_null})')
        ax.set_xlabel('Number of Samples')
        ax.set_title(f'alt_prob={alt_prob}')
        ax.grid(True)
        ax.legend()
    axes[0].set_ylabel('Power')
    plt.tight_layout()
    plt.show()


def plot_binomial_overlay_subplots(n, alt_probs, alpha_targeting, alpha_engagement, metadata_df, alpha=0.05, direction='less'):
    """
    Plot overlayed binomial distributions for null and alternative probabilities for each alt_prob as subplots,
    and highlight the critical region for the one-sided test. Label each subplot with the power.
    """
    num_probs = len(alt_probs)
    fig, axes = plt.subplots(1, num_probs, figsize=(6 * num_probs, 5), sharey=True)
    if num_probs == 1:
        axes = [axes]
    for i, alt_prob in enumerate(alt_probs):
        p_null = metadata_df.loc[(alt_prob, alpha_targeting, alpha_engagement), "conversionProbAdA_null"]
        p_alt = metadata_df.loc[(alt_prob, alpha_targeting, alpha_engagement), "conversionProbAdA"]
        x = np.arange(0, n+1)
        pmf_null = binom.pmf(x, n, p_null)
        pmf_alt = binom.pmf(x, n, p_alt)
        ax = axes[i]
        ax.plot(x, pmf_null, label=f'Null (p={p_null:.3g})', color='blue', alpha=0.7)
        ax.plot(x, pmf_alt, label=f'Alt (p={p_alt:.3g})', color='red', alpha=0.7)
        ax.fill_between(x, pmf_null, alpha=0.2, color='blue')
        ax.fill_between(x, pmf_alt, alpha=0.2, color='red')

        # Highlight the critical region for the one-sided test and compute power
        if direction == 'less':
            k_crit = int(binom.ppf(alpha, n, p_null))
            # Fill only between k_crit and pmf_null (left tail)
            ax.fill_between(x, 0, pmf_null, where=(x <= k_crit), color='orange', alpha=0.3, label=f'Rejection region (≤ {k_crit})')
            # Draw vertical line for critical point
            ax.axvline(k_crit, color='orange', linestyle='--', label=f'Critical value: {k_crit}')
            power = binom.cdf(k_crit, n, p_alt)
        else:
            k_crit = int(binom.isf(alpha, n, p_null))
            # Fill only between k_crit and pmf_null (right tail)
            ax.fill_between(x, 0, pmf_null, where=(x >= k_crit), color='orange', alpha=0.3, label=f'Rejection region (≥ {k_crit})')
            ax.axvline(k_crit, color='orange', linestyle='--', label=f'Critical value: {k_crit}')
            power = binom.sf(k_crit - 1, n, p_alt)

        ax.set_xlabel('Number of Successes')
        ax.set_title(f'alt_prob={alt_prob}\nPower={power:.3f}')
        ax.legend()
    axes[0].set_ylabel('Probability')
    plt.tight_layout()
    plt.show()
