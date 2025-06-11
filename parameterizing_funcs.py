#import opendp.prelude as dp
import numpy as np
from scipy.stats import bernoulli
import random
from adTypes import Advertisement, Conversion, Website, Campaign, features
from binomialdpy import tulap


def close(otherFeatures: features, userFeatures: features) -> float:
    if np.count_nonzero(otherFeatures) == 0:
        return 0
    return np.dot(otherFeatures, userFeatures) / np.count_nonzero(otherFeatures)

def f_engagement(adFeatures: features, userFeatures: features, alpha_engagement: float) -> bool:
    p = alpha_engagement * close(adFeatures, userFeatures)
    conversion = bernoulli.rvs(p)
    return conversion


def f_targeting(campaigns: list[Campaign], userFeatures: features, alpha_targeting: float) -> Advertisement:
    campaign = campaigns[0]
    closeA = close(campaign.adA.targetAudience, userFeatures)
    closeB = close(campaign.adB.targetAudience, userFeatures)
    if closeA > closeB:
        epsilon = closeA - closeB
        ad = random.choices([campaign.adA, campaign.adB], weights=[(1+alpha_targeting*epsilon)/2, (1-alpha_targeting*epsilon)/2])
    else:
        epsilon = closeB - closeA
        ad = random.choices([campaign.adB, campaign.adA], weights=[(1+alpha_targeting*epsilon)/2, (1-alpha_targeting*epsilon)/2])
    return ad[0]

def f_browsing(websites: list[Website], userFeatures: features) -> Website:
    closeness = [close(website.siteFeatures, userFeatures) for website in websites]
    if sum(closeness) == 0:
        return random.choice(websites)
    website = random.choices(websites, weights=closeness)
    return website[0]

def f_attribution(conversion: Conversion, conversionTime: int, adHistory: list[Advertisement]) -> Conversion:
    while conversionTime >= 0:
        if adHistory[conversionTime].campaignID == conversion.campaignID:
            conversion.attributedAd = adHistory[conversionTime]
            return conversion
    conversion.attributedAd = Advertisement(None, None, None, None)
    return conversion

    
def f_metrics(data: list[bool]) -> float: 
    return sum(data)

def f_metrics_dp_ep001(data: list[bool]) -> float:
    # Add Tulap noise
    de = 0
    b = np.exp(-0.01)
    q = 2 * de * b / (1 - b + 2 * de * b)
    noise = tulap.random(n=1, m=0, b=b, q=q)
    return sum(data) + noise

def f_metrics_dp_ep01(data: list[bool]) -> float:
    # Add Tulap noise
    de = 0
    b = np.exp(-0.1)
    q = 2 * de * b / (1 - b + 2 * de * b)
    noise = tulap.random(n=1, m=0, b=b, q=q)
    return sum(data) + noise

def f_metrics_dp_ep1(data: list[bool]) -> float:
    # Add Tulap noise
    de = 0
    b = np.exp(-0.01)
    q = 2 * de * b / (1 - b + 2 * de * b)
    noise = tulap.random(n=1, m=0, b=b, q=q)
    return sum(data) + noise
