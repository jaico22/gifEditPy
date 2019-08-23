import unittest
import cv2
import os
import pytesseract
import imutils
import numpy as np
import sys
sys.path.insert(0,'../')
import src
from src.obj.TextExtractor import TextExtractor
from src.obj.TextMatcher import TextMatcher
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import importlib 

class WordMatchingTests(unittest.TestCase): 
  
    def test_single_word_extraction(self):       
        textMatcher = TextMatcher('Good')  
        match = textMatcher.find_in_phrase('Good')
        self.assertIsNotNone(match)        
        self.assertEqual(0.0,match.startCharacterPercentage)
        self.assertEqual(1.0,match.endCharacterPercentage)
    
    def test_multi_word_extraction(self):
        text_matcher = TextMatcher('Good')
        match = text_matcher.find_in_phrase('GoodDoog')
        self.assertIsNotNone(match)
        self.assertEqual(0.0,match.startCharacterPercentage)
        self.assertEqual(0.5,match.endCharacterPercentage)
    
    def test_word_not_found_returns_none(self):
        text_matcher = TextMatcher('abcd')
        match = text_matcher.find_in_phrase('GoodDoog') 
        self.assertIsNone(match)       

if __name__ == '__main__': 
    unittest.main() 