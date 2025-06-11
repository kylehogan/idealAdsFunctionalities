from collections.abc import Callable
from adTypes import Advertisement, User, Conversion, Website, Campaign, features
import idealFunctionalities.idealSociety, idealFunctionalities.idealUserData, idealFunctionalities.idealMetrics, idealFunctionalities.idealTargeting, idealFunctionalities.idealEngagement


class AdsEcosystem:
    def __init__(self, 
                 dist, 
                 websiteLibrary: list[Website], 
                 f_targeting: Callable[[list[Campaign], features], Advertisement],
                 f_browsing: Callable[[list[Website], features], Website], 
                 f_engagement: Callable[[features, features], bool],
                 f_attribution: Callable[[Conversion, int, list[Advertisement]], Conversion],
                 f_metrics: Callable[[list[bool]], float],
                 alpha_engagement: float = 1.0,
                 alpha_targeting: float = 1.0) -> None:
        self.society = idealFunctionalities.idealSociety.Society(dist)
        self.userdata = idealFunctionalities.idealUserData.UserData()
        self.targeting = idealFunctionalities.idealTargeting.Targeting(f_targeting, alpha_targeting)
        self.engagement = idealFunctionalities.idealEngagement.Engagement(websiteLibrary, f_browsing, f_engagement, alpha_engagement)
        self.metrics = idealFunctionalities.idealMetrics.Metrics(f_attribution, f_metrics)
        self.websiteLibrary = websiteLibrary

        self.society.userdata = self.userdata
        self.targeting.userdata = self.userdata
        self.userdata.society = self.society
        self.userdata.targeting = self.targeting
        self.userdata.engagement = self.engagement
        self.engagement.userdata = self.userdata
        self.userdata.metrics = self.metrics
        self.metrics.userdata = self.userdata
    