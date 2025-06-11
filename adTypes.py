import pprint

type features = list[bool]

class Advertisement:
    def __init__(self, identifier: int, content: features, targetAudience: features, campaignID: int) -> None:
        self.identifier = identifier
        self.content = content
        self.targetAudience = targetAudience
        self.campaignID = campaignID

    def __repr__(self):
        return pprint.pformat(vars(self), indent=4)

class Campaign:
    def __init__(self, identifier: int, adA: Advertisement, adB: Advertisement, campaignAudience: features) -> None:
        self.identifier = identifier
        self.adA = adA
        self.adB = adB
        self.campaignAudience = campaignAudience

    def __repr__(self):
        return pprint.pformat(vars(self), indent=4, width=1)

#for now conversions are only click (type = 1)   
class Conversion:
    def __init__(self, campaignID: int, conversionType: bool) -> None:
        self.campaignID = campaignID
        self.conversionType = conversionType
        self.attributedAd: Advertisement

    def __repr__(self):
        return pprint.pformat(vars(self), indent=4)

class User:
    def __init__(self, identifier: int, userFeatures: features) -> None:
        self.identifier = identifier
        self.userFeatures = userFeatures

    def __eq__(self, another):
        return hasattr(another, 'identifier') and self.identifier == another.identifier

    def __hash__(self):
        return hash(self.identifier)

    def __repr__(self):
        return pprint.pformat(vars(self), indent=4) 

class Website:
    def __init__(self, identifier: int, siteFeatures: features) -> None:
        self.identifier = identifier
        self.siteFeatures = siteFeatures

    def __repr__(self):
        return pprint.pformat(vars(self), indent=4)