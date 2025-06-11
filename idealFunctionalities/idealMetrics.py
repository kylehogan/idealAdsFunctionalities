from collections.abc import Callable
from collections import defaultdict
from adTypes import Advertisement, Conversion, Campaign
import idealFunctionalities.idealUserData

class Metrics:
    def __init__(self, f_attribution: Callable[[Conversion, int, list[Advertisement]], Conversion], f_metrics: Callable[[list[bool]], float]) -> None:
        self.f_attribution = f_attribution
        self.f_metrics = f_metrics
        self.attributedAds: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(lambda: [], {}), {})
        self.userdata: idealFunctionalities.idealUserData.UserData

    def attribution(self, userID: int) -> None:
        conversion, conversionIndex, adsHistory = self.userdata.attributionData(userID)
        conversion = self.f_attribution(conversion, conversionIndex, adsHistory)
        self.attributedAds[conversion.campaignID][conversion.attributedAd.identifier].append(conversion)
        return
    

    def metrics(self, campaign: Campaign) -> tuple[float, float]:
        if campaign.identifier not in self.attributedAds:
            return False
        #grab all the conversions that arent a view
        adAconversions = [conversion.conversionType for conversion in self.attributedAds[campaign.identifier][campaign.adA.identifier] if conversion.conversionType]
        adAconversions = self.f_metrics(adAconversions)
        adBconversions = [conversion.conversionType for conversion in self.attributedAds[campaign.identifier][campaign.adB.identifier] if conversion.conversionType]
        adBconversions = self.f_metrics(adBconversions)
        return adAconversions, adBconversions
