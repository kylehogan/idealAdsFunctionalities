from collections.abc import Callable
from adTypes import Advertisement, Campaign, features
import idealFunctionalities.idealUserData

class Targeting:
    def __init__(self, f_targeting: Callable[[list[Campaign], features], Advertisement], alpha_targeting: float) -> None:
        self.campaigns: list[Campaign] = []
        self.f_targeting = f_targeting
        self.userdata: idealFunctionalities.idealUserData.UserData
        self.alpha_targeting = alpha_targeting

    def registerCampaign(self, campaign: Campaign) -> bool:
        self.campaigns.append(campaign)
        return True
    
    def targetAds(self, userID: int) -> bool:
        targetingData = self.userdata.adTargetingData(userID)
        if targetingData is not False:
            if userID in targetingData:
                userFeatuers = targetingData[userID]["userFeatures"]
                context = targetingData[userID]["context"]
                ad = self.f_targeting(self.campaigns, userFeatuers, self.alpha_targeting)
                success = self.userdata.adTargetingDisplay(userID, ad, context.identifier)
                return success
        return False
