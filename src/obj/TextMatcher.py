from src.obj.CharacterMatch import CharacterMatch

class TextMatcher :
    def __init__(self,target_phrase) : 
        self.target_phrase = target_phrase
        
    def find_in_phrase(self,input_phrase) : 
        """ 
        Finds target string in phrase. 
    
        Retruns an object containing positional information about the starting and
        ending positions of the registered target phrase in the input phrase
    
        Parameters: 
        arg1 (string): Phrase to search for registered target phrase in
    
        Returns: 
        CharacterMatch: Match metadata related to the position of the input
        phrase in the registered target phrase
    
        """       
        phrase_pos = str(input_phrase).find(self.target_phrase)

        if phrase_pos != -1 : 
            startCharacterPercentage = phrase_pos / len(input_phrase)
            endCharacterPercentage = (phrase_pos + len(self.target_phrase)) / len(input_phrase)
            return CharacterMatch(self.target_phrase,startCharacterPercentage,endCharacterPercentage)
        
        return None
