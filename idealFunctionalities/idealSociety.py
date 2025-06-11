import random
from adTypes import User
import idealFunctionalities.idealUserData

class Society:
    def __init__(self, dist) -> None:
        self.dist = dist
        self.userCount: int = 0
        self.userdata: idealFunctionalities.idealUserData.UserData


    def newUser(self, userID: int) -> User:
        self.userCount += 1
        userFeatures = self.dist.pop()  
        user = User(userID, userFeatures)
        return user
