class CharacterMatch : 
    def __init__(self,word,startCharacterPercentage,endCharacterPercentage):
        self.word = word
        self.startCharacterPercentage = startCharacterPercentage
        self.endCharacterPercentage = endCharacterPercentage
    
    def __str__(self) : 
        return self.word + " found at " + str(self.startCharacterPercentage) + \
            "% to "  + str(self.endCharacterPercentage) + "% of input phrase"