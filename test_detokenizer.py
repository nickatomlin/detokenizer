import unittest
from detokenizer import GPT2Detokenizer  # Replace with your actual module import

class TestGPT2Detokenizer(unittest.TestCase):

    def setUp(self):
        self.detokenizer = GPT2Detokenizer()

    def test_detokenize(self):
        tokens = ["She", "said", ",", "``", "I", "love", "Python", "programming", "''", "."]
        self.assertEqual(self.detokenizer.detokenize(tokens), "She said, ``I love Python programming''.")

        tokens = ["It", "'s", "a", "beautiful", "day", "!"]
        self.assertEqual(self.detokenizer.detokenize(tokens), "It's a beautiful day!")

        tokens = ["John", "'s", "book", "is", "``", "The", "Art", "of", "Computer", "Programming", "''", "."]
        self.assertEqual(self.detokenizer.detokenize(tokens), "John's book is ``The Art of Computer Programming''.")

if __name__ == '__main__':
    unittest.main()
