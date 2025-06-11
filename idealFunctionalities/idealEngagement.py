from collections.abc import Callable
from adTypes import Conversion, Website, features
import idealFunctionalities.idealUserData

class Engagement:
    def __init__(
            self, websiteLibrary: list[Website], 
            f_browsing: Callable[[list[Website], features], Website], 
            f_engagement: Callable[[features, features], bool],
            alpha_engagement: float) -> None:
        
        self.f_browsing = f_browsing
        self.f_engagement = f_engagement
        self.alpha_engagement = alpha_engagement
        self.userdata: idealFunctionalities.idealUserData.UserData
        self.websiteLibrary = websiteLibrary

    def browsing(self, userID: int) -> bool:
        user = self.userdata.browsingData(userID)
        website = self.f_browsing(self.websiteLibrary, user.userFeatures)
        success = self.userdata.browsingDisplay(user.identifier, website)
        return success
    
    def engagement(self, userID: int) -> bool:
        engagementData = self.userdata.engagementData(userID)
        if engagementData is not False:
            if userID in engagementData:
                userFeatuers = engagementData[userID]["userFeatures"]
                #context = engagementData[userID]["context"].siteFeatures
                ad = engagementData[userID]["ad"]
                conversion = Conversion(ad.campaignID, self.f_engagement(ad.content, userFeatuers, self.alpha_engagement))
                success = self.userdata.recordConversion(userID, ad.identifier, conversion)
                return success
        return False
