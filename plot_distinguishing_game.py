import matplotlib.pyplot as plt
import pandas as pd

def plotNumSamples(pval_df, metadata_df, direction='left', combo=[], st_dev=True, null_pr=0.5, directory='plots/', plot_type='private_v_nonprivate'):
    # Initialize a figure
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.size'] = 18

    colors = ["#e27c7c", "#a86464", '#8B5757', "#6d4b4b", "#503f3f"]
    markers = ['o', 's', 'd', '^', 'D']
    match plot_type:
        case 'private_v_nonprivate':
            colors = ["#e27c7c", "#a86464", "#6d4b4b", "#503f3f"]
            linetype = ['Private', 'Non-Private', 'Baseline'] 
            labels = [f'{linetype} (ε={"N/A" if epsilon == 0 else epsilon}, α_targeting={alpha_targeting}, α_engagement={alpha_engagement})' for (alpha_targeting, alpha_engagement, epsilon), linetype in zip(combo, linetype)]
        case 'targeting':
            labels = [f'α_targeting={alpha}' for alpha, _, _ in combo]
        case 'epsilon':
            labels = [f'ε={"N/A" if epsilon == 0 else epsilon}' for _, _, epsilon in combo]
        case 'engagement':
            labels = [f'α_engagement={alpha}' for _, alpha, _ in combo]
        case 'test':
            labels = [f'α_targeting={alpha_targeting}, α_engagement={alpha_engagement}, ε={"N/A" if epsilon == 0 else epsilon}' for alpha_targeting, alpha_engagement, epsilon in combo]

    for entry in combo:
        alpha_targeting = entry[0]
        alpha_engagement = entry[1]
        epsilon = entry[2]
        results = pval_df.loc[(alpha_targeting, alpha_engagement, epsilon)]
        means = results.mean(axis=0)

        # Plot private mean and standard deviation for each alpha
        plt_color = colors.pop(0)
        plt.plot(
            means.index, 
            means, 
            marker=markers.pop(0), 
            label=labels.pop(0), 
            color=plt_color, 
            linewidth=2.5,  # Make lines thicker
            markersize=10   # Make points larger
        )
        if st_dev:
            private_stds = results.std(axis=0)
            plt.fill_between(
                means.index,
                means - private_stds,
                means + private_stds,
                color=plt_color,
                alpha=0.2,
                label=f'Private Std Dev (ε={"N/A" if epsilon == 0 else epsilon}, α_targeting={alpha_targeting}, α_engagement={alpha_engagement})'
            )
    # Set x-ticks to alt_probs and label with TVD from metadata_df
    alt_probs = means.index
    # Get TVD values for the first alpha_targeting/alpha_engagement in combo (assumes TVD is the same for all combos at each alt_prob)
    tvd_labels = [f"{metadata_df.loc[(alt_prob, combo[0][0], combo[0][1]), 'tvd']:.3f}" for alt_prob in alt_probs]
    plt.xticks(ticks=alt_probs, labels=tvd_labels, rotation=45)
    plt.xlabel(f'Total Variation Distance (TVD) between D0 and D1')

    plt.ylabel('Number of Samples')
    plt.title(f'Campaign Size Required to Distinguish') 
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Show the plot
    plt.savefig(f'{directory}/pval_{direction}_{plot_type}.svg', format='svg')
    plt.savefig(f'{directory}/pval_{direction}_{plot_type}.png', format='png', dpi=360)
    plt.show()
