import matplotlib.pyplot as plt
import pandas as pd

def plotNumSamples(pval_df, metadata_df, direction='left', combo=[], st_dev = True, null_pr=0.5, directory='plots/', plot_type='private_v_nonprivate'):
    # Initialize a figure
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.size'] = 16

    colors = ["#e27c7c", "#a86464", '#8B5757', "#6d4b4b", "#503f3f"]
    markers = ['o', 's', 'd', '^', 'D']
    match plot_type:
        case 'private_v_nonprivate':
            colors = ["#e27c7c", "#a86464", "#6d4b4b", "#503f3f"]
            linetype = ['Private', 'Non-Private', 'Baseline'] 
            labels = [f'{linetype} (ε={epsilon}, α_targeting={alpha_targeting}, α_engagement={alpha_engagement})' for (alpha_targeting, alpha_engagement, epsilon), linetype in zip(combo, linetype)]
        case 'targeting':
            labels = [f'α_targeting={alpha}' for alpha, _, _ in combo]
        case 'epsilon':
            labels = [f'ε={epsilon}' for _, _, epsilon in combo]
        case 'engagement':
            labels = [f'α_engagement={alpha}' for _, alpha, _ in combo]
        case 'test':
            labels = [f'α_targeting={alpha_targeting}, α_engagement={alpha_engagement}, ε={epsilon}' for alpha_targeting, alpha_engagement, epsilon in combo]

    for entry in combo:
        alpha_targeting = entry[0]
        alpha_engagement=entry[1]
        epsilon=entry[2]
        results = pval_df.loc[(alpha_targeting, alpha_engagement, epsilon)]
        means = results.mean(axis=0)

        # Plot private mean and standard deviation for each alpha
        plt_color = colors.pop(0)
        plt.plot(means.index, means, marker=markers.pop(0), label=labels.pop(0), color=plt_color)
        if st_dev:
            private_stds = results.std(axis=0)
            plt.fill_between(
                means.index,
                means - private_stds,
                means + private_stds,
                color=plt_color,
                alpha=0.2,
                label=f'Private Std Dev (ε={epsilon}, α_targeting={alpha_targeting}, α_engagement={alpha_engagement})'
            )
    # Set x-ticks to alt_probs and label with TVD from metadata_df
    alt_probs = means.index
    # Get TVD values for the first alpha_targeting/alpha_engagement in combo (assumes TVD is the same for all combos at each alt_prob)
    tvd_labels = [f"{metadata_df.loc[(alt_prob, combo[0][0], combo[0][1]), 'tvd']:.3f}" for alt_prob in alt_probs]
    plt.xticks(ticks=alt_probs, labels=tvd_labels)
    plt.xlabel(f'Total Variation Distance (TVD) between D0 and D1')

    plt.ylabel('Number of Samples')
    plt.title(f'Campaign Size Required to Distinguish') 
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Show the plot
    plt.savefig(f'{directory}/pval_{direction}_{plot_type}.svg', format='svg')
    plt.show()


def visualize_sample_complexity_df(clicks_df, plot_type='private_v_nonprivate'):
    print(f"Max value across entire DataFrame: {clicks_df.max().max()}")
    print(f"NaN count per column in clicks_df: {clicks_df.isna().sum()}")
    if plot_type == 'test':
        print(f'count of values less than 20 {(clicks_df < 20).sum()}')  #count of values less than 20
        print(f'count of values less than 100 {(clicks_df < 100).sum()}') 
        print(f'shape of each column: {[clicks_df[col].shape for col in clicks_df.columns]}') #shape of each column for (0.1,1,0)
        clicks_df.hist(bins=100, figsize=(10, 6))
        plt.show()
        clicks_df[clicks_df < 200].hist(bins=50, figsize=(10, 6))
        plt.show()
    elif plot_type == 'private_v_nonprivate':
        print(f'count of values less than 20 {(clicks_df.loc[(0.6,0.2,0.1)] < 20).sum()}')
        print(f'count of values less than 100 {(clicks_df.loc[(0.6,0.2,0.1)] < 100).sum()}')
        print(f'shape of each column: {[clicks_df.loc[(0.6,0.2,0.1)][col].shape for col in clicks_df.columns]}') #shape of each column for (0.1,1,0)
        clicks_df.loc[(0.6,0.2,0.1)].hist(bins=100, figsize=(10, 6))
        plt.show()
        clicks_df[clicks_df<500].loc[(0.6,0.2,0.1)].hist(bins=100, figsize=(10, 6))
        plt.show()