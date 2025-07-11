import bindata as bnd
import numpy as np
import itertools
from parameterizing_funcs import f_attribution, f_targeting, f_browsing, f_engagement, f_metrics, close, f_metrics_dp_ep001, f_metrics_dp_ep01, f_metrics_dp_ep1
from adTypes import Advertisement, Website, Campaign
from idealFunctionalities.idealAdsEcosystem import AdsEcosystem
from binomialdpy.pvalue import left as pvalue_left
from binomialdpy.pvalue import right as pvalue_right
from scipy.stats import binomtest, binom
import pandas as pd
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import matplotlib.pyplot as plt
import argparse
from os import makedirs, path, listdir
from collections import defaultdict
from binomial_plots import plot_binomial_power_curve_subplots, plot_binomial_overlay_subplots
from plot_distinguishing_game import visualize_sample_complexity_df, plotNumSamples
import shutil
from sequential_distinguishing_game import produceMetadataDataframe

def produceSampleComplexity(campaign, adA, adB, website1, website2,
                                            metadata_df = None,
                                            filename='', 
                                             trials=[], 
                                             campaign_size=10, 
                                             null_prob=0.1, 
                                             alt_probs = [0.25, 0.5, 0.75], 
                                             alpha_targeting_values = [0.1,0.5,0.9,1], 
                                             alpha_engagement_values = [0.1,0.5,0.9,1],
                                             epsilons = [(0,f_metrics), (0.01,f_metrics_dp_ep001), (0.1,f_metrics_dp_ep01), (1,f_metrics_dp_ep1)],
                                             delta=0,
                                             direction='left',
                                             progress_callback=None):

    metadata_df = metadata_df

    corr = np.array([[1., -0.25, -0.0625],
                    [-0.25,   1.,  0.25],
                    [-0.0625, 0.25, 1.]])
    n = 100000 #for pr dist of users

    array_multiindex = pd.MultiIndex.from_arrays(
            [alpha_targeting_values,
                alpha_engagement_values,
                [ep[0] for ep in epsilons]], 
                names=['alpha_targeting', 'alpha_engagement', 'epsilon'])

    #add the trials to index without creating all combinations of alpha targeting, engagement, and epsilon
    multiindex = pd.MultiIndex.from_frame(array_multiindex.to_frame().merge(pd.Series(trials, name="trial"), how='cross'))

    #allow for adding new data or overwriting only some data without recomputing everything
    if path.exists(filename):
        pval_df = pd.read_parquet(filename, engine='pyarrow')
        # Ensure the index is the union of the current index and multiindex
        pval_df = pval_df.reindex(pval_df.index.union(multiindex, sort=True))
        # Add new columns for any alt_prob not already present
        for alt_prob in alt_probs:
            if alt_prob not in pval_df.columns:
                pval_df[alt_prob] = pd.NA
    else:
        # Initialize DataFrames with the updated multiindex
        pval_df = pd.DataFrame(index=multiindex, columns=alt_probs, dtype=int)
        pval_df.sort_index(inplace=True)

    combo = list(zip(alpha_targeting_values,alpha_engagement_values,epsilons))

    for alt_prob in alt_probs:
        margprobEco2 = [0.2, 0.5, alt_prob]

        commonprob_eco2 = bnd.bincorr2commonprob(margprob=margprobEco2, bincorr=corr)

        user_dist_eco2 = bnd.rmvbin(margprob=np.diag(commonprob_eco2), 
                                commonprob=commonprob_eco2, N=n)

        unique_users, counts = np.unique(user_dist_eco2, axis=0, return_counts=True)
        userProb_eco2 = defaultdict(float)
        userProb_eco2.update(dict(zip(map(tuple, unique_users), counts/n)))

        for alpha_targeting, alpha_engagement, epsilon in combo:
            #empirical observations for clicks table
            for trial in trials:
                #resample the users for each trial
                user_dist_eco2 = bnd.rmvbin(margprob=np.diag(commonprob_eco2), 
                                commonprob=commonprob_eco2, N=campaign_size)

                ecosystem2 = AdsEcosystem(
                        dist = user_dist_eco2.tolist(),
                        websiteLibrary=[website1, website2] , 
                        f_targeting=f_targeting,
                        alpha_targeting=alpha_targeting, 
                        f_browsing=f_browsing, 
                        f_engagement=f_engagement,
                        alpha_engagement=alpha_engagement, 
                        f_attribution=f_attribution,
                        f_metrics=epsilon[1])
                    
                ecosystem2.targeting.registerCampaign(campaign)

                for userID in range(campaign_size):
                    ecosystem2.engagement.browsing(userID=userID)
                    ecosystem2.targeting.targetAds(userID=userID)
                    ecosystem2.engagement.engagement(userID=userID)
                    ecosystem2.metrics.attribution(userID=userID)
                    state = random.getstate() #grab state before setting it for metrics
                    statenp = np.random.get_state()
                    random.seed(int(trial)) #set state for metrics
                    np.random.seed(trial)
                    clicks, _ = ecosystem2.metrics.metrics(campaign=campaign) #metrics should use the same randomness per trial for dp noise
                    random.setstate(state) #reset state after metrics
                    np.random.set_state(statenp)

                    p_alt = metadata_df.loc[(alt_prob, alpha_targeting, alpha_engagement), "conversionProbAdA"]
                    p_null = metadata_df.loc[(alt_prob, alpha_targeting, alpha_engagement), "conversionProbAdA_null"]

                    if epsilon[0] != 0:
                        # Add Tulap noise for the current epsilon
                        de = delta
                        b = np.exp(-epsilon[0])
                        q = 2 * de * b / (1 - b + 2 * de * b)
                        if direction == 'left':
                            pval = pvalue_left(clicks, n=userID+1, p=null_prob, b=b, q=q)[0]
                        else:
                            pval = pvalue_right(clicks, n=userID+1, p=null_prob, b=b, q=q)[0]
                    else:
                        if direction == 'left':
                            pval = binomtest(k=int(clicks), n=userID+1, p=p_null, alternative='less').pvalue
                        else:
                            pval = binomtest(k=int(clicks), n=userID+1, p=p_null, alternative='greater').pvalue
                    
                    # if plot_type == "test":
                    #     print("clicks: ", clicks, "userID: ", userID, "pval: ", pval, "power: ", power, "alt_prob: ", p_alt, "null pr: ", p_null)

                    if pval <= 0.05 and clicks > 0:
                        #print("FINISHED: clicks: ", clicks, "userID: ", userID, "pval: ", pval, "power: ", power, "alt_prob: ", p_alt, "null_prob: ", p_null)

                        # if plot_type == "test" and clicks > 0 and clicks < (userID + 1):
                        #     pval_noep = binomtest(k=int(clicks), n=userID+1, p=metadata_df.loc[(alt_prob, alpha_targeting, alpha_engagement), "conversionProbAdA_null"], alternative='less').pvalue 
                        #     print("FINISHED: clicks: ", clicks, "userID: ", userID, "pval: ", pval, "pval_noep: ", pval_noep, "alt_prob: ", metadata_df.loc[(alt_prob, alpha_targeting, alpha_engagement), "conversionProbAdA"], "null pr: ", metadata_df.loc[(alt_prob, alpha_targeting, alpha_engagement), "conversionProbAdA_null"])
                        pval_df.loc[(alpha_targeting, alpha_engagement, epsilon[0], trial), alt_prob] = userID
                        break
                if progress_callback:
                    progress_callback()

    pval_df.to_parquet(filename, engine='pyarrow', compression='snappy')


def combinePValDf(filename='', filename_metadata='', alt_probs = [], trial_subsets=[], plot_type='private_v_nonprivate', directory='plots/', campaign_size=200000):
    # Concatenate all found files
    altprob_dfs = []
    #combine all trials for each alt_prob
    for alt_prob in alt_probs:
        print(f"Combining data for alt_prob: {alt_prob}")
        combined_df = pd.concat([pd.read_parquet(f"{directory}/{f}") for f in listdir(directory) if f.endswith('.parquet') and "combined" not in f and f"altprob{alt_prob}_" in f],
            axis=0,  # Combine along the rows (index)
            verify_integrity=True
        )
        combined_df.sort_index(inplace=True)
        altprob_dfs.append(combined_df)
    
    #combine all alt_probs into one df
    clicks_df = pd.concat(altprob_dfs, axis=1, verify_integrity=True)

    metadata_dfs = [pd.read_parquet(f"{directory}/metadata/{f}") for f in listdir(f"{directory}/metadata") if f.endswith('.parquet') and "combined" not in f]
    metadata_df = pd.concat(metadata_dfs, verify_integrity=True)

    filename = f'{directory}/combined.parquet'
    filename_metadata = f'{directory}/metadata/combined_metadata.parquet'
    clicks_df.to_parquet(filename, engine='pyarrow', compression='snappy')
    metadata_df.to_parquet(filename_metadata, engine='pyarrow', compression='snappy')
    return clicks_df, metadata_df

def process_chunk(trial_chunk, alt_prob, campaign, adA, adB, website1, website2, metadata_dataframe, campaign_size, null_prob, alpha_targeting_values, alpha_engagement_values, epsilons, direction, filename):
    """
    Process a single chunk of trials for a given alt_prob.
    """
    with tqdm(total=len(trial_chunk) * len(alpha_targeting_values), desc=f"alt_prob={alt_prob}, trials={trial_chunk[0]}-{trial_chunk[-1]}", position=multiprocessing.current_process()._identity[0], leave=False) as pbar:
        def progress_callback(*args, **kwargs):
            pbar.update(1)

        produceSampleComplexity(
            campaign=campaign,
            adA=adA,
            adB=adB,
            website1=website1,
            website2=website2,
            metadata_df = metadata_dataframe,
            trials=list(trial_chunk),
            campaign_size=campaign_size,
            filename=filename,
            null_prob=null_prob,
            alt_probs=[alt_prob],
            alpha_targeting_values=alpha_targeting_values,
            alpha_engagement_values=alpha_engagement_values,
            epsilons=epsilons,
            direction=direction,
            progress_callback=progress_callback
        )
    return alt_prob, len(trial_chunk)

def runParallelSampleProductionByTrials(campaign, adA, adB, website1, website2,
                                        trials=0,
                                        trial_start=0, 
                                        campaign_size=0, 
                                        null_prob=0.5, 
                                        alpha_targeting_values=[], 
                                        alpha_engagement_values=[], 
                                        epsilons=[], 
                                        direction='left', 
                                        alt_probs=[], 
                                        num_processes=8,
                                        num_chunks=16,
                                        filename='',
                                        filename_metadata = ''):
    # Divide trials into 20 fixed chunks
    trial_chunks = np.array_split(range(trial_start, trial_start+trials), num_chunks)

    # Initialize progress bars for each alt_prob
    progress_bars = {
        alt_prob: tqdm(total=trials, desc=f"alt_prob={alt_prob}", position=i, leave=False)
        for i, alt_prob in enumerate(alt_probs)
    }

    # Function to update progress bars
    def update_progress(alt_prob, completed_trials):
        progress_bars[alt_prob].update(completed_trials)

    # Use ProcessPoolExecutor to process chunks dynamically
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for alt_prob in alt_probs:
            metadata_df = produceMetadataDataframe(adA, adB,
                                                null_prob=null_prob, 
                                                alt_probs=[alt_prob], 
                                                alpha_targeting_values=alpha_targeting_values, 
                                                alpha_engagement_values=alpha_engagement_values,
                                                epsilons=epsilons)
            metadata_df.to_parquet(filename_metadata.replace("altprob_trial_subset", f"altprob{alt_prob}"), engine='pyarrow', compression='snappy')
            for trial_chunk in trial_chunks:
                filename_chunk = filename.replace("altprob_trial_subset", f"altprob{alt_prob}_trial_subset{trial_chunk[0]}_{trial_chunk[-1]}")
                futures.append(
                    executor.submit(
                        process_chunk,
                        trial_chunk,
                        alt_prob,
                        campaign,
                        adA,
                        adB,
                        website1,
                        website2,
                        metadata_df,
                        campaign_size,
                        null_prob,
                        alpha_targeting_values,
                        alpha_engagement_values,
                        epsilons,
                        direction,
                        filename_chunk
                    )
                )

        # Wait for all tasks to complete and update progress bars
        for future in tqdm(as_completed(futures), desc="Processing all chunks", leave=False):
            alt_prob, completed_trials = future.result()
            update_progress(alt_prob, completed_trials)

    # Close all progress bars
    for pbar in progress_bars.values():
        pbar.close()

if __name__ == "__main__":

    adA = Advertisement(identifier=3, content=[1,0,1], targetAudience=[1,1,1], campaignID=5)
    adB = Advertisement(identifier=8, content=[1,0,0], targetAudience=[1,1,0], campaignID=5)

    campaign = Campaign(identifier=5, adA=adA, adB=adB, campaignAudience=[1,1,0])

    #we aren't using the context for targeting and engagement right now
    website1 = Website(identifier=15, siteFeatures=[1,0,1])
    website2 = Website(identifier=30, siteFeatures=[1,1,0])

    parser = argparse.ArgumentParser(description="Produce samples and plot complexity")
    parser.add_argument("--plot_type", help="private_v_nonprivate, targeting, epsilon, or engagement", default='private_v_nonprivate')
    parser.add_argument("--alt_probs", help="marginal probabilities for D1 (D0 has pr=0.9). format as a list of floats like '[0.5,0.6,0.7,0.8]'", default= '[0.5,0.6,0.7,0.8,0.825,0.85,0.875]', type=lambda s: [float(item) for item in s.strip('[]').split(',')])
    parser.add_argument("--trials", help="number of trials to run", default=500, type=int)
    parser.add_argument("--cores", help="number of trials to run", default=8, type=int)
    parser.add_argument("--plots_only", help="only produce plots and not samples", action='store_true')
    parser.add_argument("--show_st_dev", help="show standard deviation on plot", action='store_true')
    parser.add_argument("--trial_start", help="where to start the trial index (default 0)", default=0, type=int)
    parser.add_argument("--num_chunks", help="number of chunks to divide trials into for parallel processing (default 16)", default=16, type=int)
    parser.add_argument("--null_prob", help="null marginal probability (default 0.9)", default=0.9, type=float)
    parser.add_argument("--clean", help="remove previous data for plot type", action='store_true')

    args = parser.parse_args()

    # Define the path for the new folder
    new_folder_path = path.join('plots/', args.plot_type)

    # Create the folder
    if args.clean and path.exists(new_folder_path) and not args.plots_only: #cant remove and plot. prevent accidents.
        shutil.rmtree(new_folder_path)

    try:
        makedirs(new_folder_path, exist_ok=True)
        makedirs(f'{new_folder_path}/metadata', exist_ok=True)
    except OSError as e:
        print(f"Error creating folder: {e}")

    campaign_size = 100000
    trials = args.trials
    cores=args.cores
    num_chunks = args.num_chunks
    alt_probs = args.alt_probs  # Marginal probabilities for the test bit
    null_prob = args.null_prob  # Null marginal probability for the test bit
    direction = 'left' if null_prob > max(alt_probs) else 'right'
    trial_start = args.trial_start

    plot_type = args.plot_type

    filename = f'{new_folder_path}/nullprob_{null_prob}_altprob_trial_subset_{plot_type}.parquet'
    filename_metadata = f'{new_folder_path}/metadata/nullprob_{null_prob}_altprob_trial_subset_{plot_type}.parquet'
            

    match plot_type:
        case 'private_v_nonprivate':    
            #make a plot for game: realistic non-private, realistic private, baseline
            alpha_targeting_values = [0.6, 0.9, 1]
            alpha_engagement_values = [0.2, 0.2, 1]   
            epsilons = [(0.1,f_metrics_dp_ep01), (0,f_metrics), (0,f_metrics)]
            # filename = f'{new_folder_path}/pval_{direction}_altprob_trial_subset_private_v_nonprivate.parquet'
            # filename_metadata = f'{new_folder_path}/pval_{direction}_altprob_trial_subset_private_v_nonprivate_metadata.parquet'
        case 'targeting':
            #make a plot for game: vary only alpha-targeting
            alpha_targeting_values = [0.1, 0.5, 0.9, 1]
            alpha_engagement_values = [1] * len(alpha_targeting_values)  
            epsilons = [(0,f_metrics)] * len(alpha_targeting_values)
        case 'epsilon':
            #make a plot for game: vary only epsilon
            epsilons = [(0,f_metrics), (0.01,f_metrics_dp_ep001), (0.1,f_metrics_dp_ep01), (1,f_metrics_dp_ep1)]
            alpha_targeting_values = [1] * len(epsilons)
            alpha_engagement_values = [1] * len(epsilons)
        case 'engagement':  
            #make a plot for game: vary only alpha-engagement
            alpha_engagement_values = [0.1, 0.5, 0.9, 1]
            #alpha_engagement_values = [0.1]
            alpha_targeting_values = [1] * len(alpha_engagement_values)
            epsilons = [(0,f_metrics)] * len(alpha_engagement_values)
        case "test":
            alpha_targeting_values = [1]
            alpha_engagement_values = [1]
            epsilons = [(0,f_metrics)]

    if not args.plots_only:
        runParallelSampleProductionByTrials(campaign, adA, adB, website1, website2, 
                                    trials=trials,
                                    trial_start=trial_start, 
                                    campaign_size=campaign_size, 
                                    null_prob=null_prob, 
                                    alpha_targeting_values=alpha_targeting_values,
                                    alpha_engagement_values=alpha_engagement_values, 
                                    epsilons=epsilons,
                                    direction=direction,
                                    alt_probs=alt_probs,
                                    num_processes=cores,
                                    num_chunks=num_chunks,
                                    filename=filename,
                                    filename_metadata = filename_metadata)

    clicks_df, metadata_df = combinePValDf(filename=filename, filename_metadata=filename_metadata, alt_probs=alt_probs, trial_subsets=np.array_split(range(trials), num_chunks), plot_type=plot_type, directory=new_folder_path, campaign_size=campaign_size)
    # plotNumSamples(clicks_df, metadata_df, combo=[(alpha_targeting_values[i], alpha_engagement_values[i], epsilons[i][0]) for i in range(len(epsilons))], st_dev=args.show_st_dev, null_pr=null_prob, directory=new_folder_path, plot_type=plot_type)

    # plot_binomial_overlay_subplots(
    #     n=100,
    #     alt_probs=alt_probs,
    #     alpha_targeting=alpha_targeting_values[0],
    #     alpha_engagement=alpha_engagement_values[0],
    #     metadata_df=metadata_df
    # )

    
    # plot_binomial_power_curve_subplots(alt_probs=alt_probs, 
    #                                     alpha=0.05, 
    #                                     max_samples=10,
    #                                     alpha_targeting=alpha_targeting_values[0],
    #                                     alpha_engagement=alpha_engagement_values[0],
    #                                     metadata_df=metadata_df)
    
    # visualize_sample_complexity_df(clicks_df, plot_type=plot_type)
    #type two error is the amonut of the alt distribution thats inside the non-rejection region of the null distribution