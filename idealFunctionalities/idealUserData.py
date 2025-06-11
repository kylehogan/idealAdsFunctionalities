from collections import defaultdict
import pprint
from adTypes import Advertisement, User, Conversion, Website
import idealFunctionalities.idealSociety, idealFunctionalities.idealEngagement, idealFunctionalities.idealMetrics, idealFunctionalities.idealTargeting

class UserData:
    def __init__(self) -> None:
        self.activeUsers: dict[User.identifier, self.__BrowsingHistory] = {}
        self.targeting: idealFunctionalities.idealTargeting.Targeting
        self.society: idealFunctionalities.idealSociety.Society
        self.engagement: idealFunctionalities.idealEngagement.Engagement
        self.metrics: idealFunctionalities.idealMetrics.Metrics
        self.conversionLogAdA = [] #bookkeeping
    
    def __repr__(self):
        return pprint.pformat(vars(self), indent=4, width=1)

    #create a new version of the user for userData to work with
    class __BrowsingHistory(User):
        def __init__(self, user: User) -> None:
            super().__init__(user.identifier, user.userFeatures)
            self.userFeatures = user.userFeatures
            self.browsingHistory: list[(Website, Advertisement, Conversion)] = []
            self.conversionCount: dict[int, int] = defaultdict(lambda: 0, {}) #bookkeeping
            self.targetingCount: dict[int, int] = defaultdict(lambda: 0, {}) #bookkeeping
            self.targetIndex: int = 0
            self.engagementIndex: int = 0
            self.attributionIndex: int = 0

    def browsingData(self, userID: int) -> User:
        if userID not in self.activeUsers:
            newUser = self.society.newUser(userID=userID)
            self.activeUsers[newUser.identifier] = self.__BrowsingHistory(newUser)
        user = self.activeUsers[userID]
        return user
    
    def engagementData(self, userID: int) -> bool | dict:
        #trying to engage with an ad that doesn't exist yet
        if userID not in self.activeUsers or self.activeUsers[userID].engagementIndex >= self.activeUsers[userID].targetIndex:
            return False
        return {userID: {
                        "userFeatures": self.activeUsers[userID].userFeatures, 
                        "context": self.activeUsers[userID].browsingHistory[self.activeUsers[userID].engagementIndex][0], 
                        "ad": self.activeUsers[userID].browsingHistory[self.activeUsers[userID].engagementIndex][1]
                        }}

    def recordConversion(self, userID: int, adID: int, conversion: Conversion) -> bool:
        #check that user exists and that the engagement is for the current ad
        if userID not in self.activeUsers or adID != self.activeUsers[userID].browsingHistory[self.activeUsers[userID].engagementIndex][1].identifier:
            return False
        else:
            user = self.activeUsers[userID]
            user.browsingHistory[user.engagementIndex] = (user.browsingHistory[user.engagementIndex][0], user.browsingHistory[user.engagementIndex][1], conversion)
            user.engagementIndex += 1
            user.conversionCount[adID] += conversion.conversionType
            #bookkeeping
            if adID == 3:
                self.conversionLogAdA.append(conversion.conversionType)
            else: 
                self.conversionLogAdA.append(0)

            self.activeUsers[userID] = user
            return True

    def browsingDisplay(self, userID: int, website: Website) -> bool:
        if userID not in self.activeUsers:
            return False
        else:
            user = self.activeUsers[userID]
            user.browsingHistory.append((website, None, None))
            self.activeUsers[userID] = user 
            return True

    def adTargetingData(self, userID: int) -> bool | dict:
        #trying to target an ad past where the user has browsed
        if userID not in self.activeUsers or self.activeUsers[userID].targetIndex >= len(self.activeUsers[userID].browsingHistory):
            return False
        return {userID: {
                        "userFeatures": self.activeUsers[userID].userFeatures, 
                        "context": self.activeUsers[userID].browsingHistory[self.activeUsers[userID].targetIndex][0]
                        }}
    
    def adTargetingDisplay(self, userID: int, ad: Advertisement, websiteidentifier: int) -> bool:
        #check that user exists and that the site being targeted is the current one
        if userID not in self.activeUsers or websiteidentifier != self.activeUsers[userID].browsingHistory[self.activeUsers[userID].targetIndex][0].identifier:
            return False
        else:
            user = self.activeUsers[userID]
            user.browsingHistory[user.targetIndex] = (user.browsingHistory[user.targetIndex][0], ad, None)
            user.targetingCount[ad.identifier] += 1
            user.targetIndex += 1
            self.activeUsers[userID] = user
            return True
        
    def attributionData(self, userID: int) -> tuple[Conversion, int, list[Advertisement]]:
        #check that user exists and there exists a conversion at the attribution index 
        if userID not in self.activeUsers or type(self.activeUsers[userID].browsingHistory[self.activeUsers[userID].attributionIndex][2]) != Conversion:
            return False
        user = self.activeUsers[userID]        
        conversion = user.browsingHistory[user.attributionIndex][2]
        conversionIndex = user.attributionIndex
        #check slicing -- only want up to the attribution index
        adsHistory = [item[1] for item in user.browsingHistory[:(user.attributionIndex+1)]]
        self.activeUsers[userID].attributionIndex += 1
        return conversion, conversionIndex, adsHistory
