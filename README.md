# GPT-2 Based Detokenizer

Notice: this repository was almost entirely written with the assistance of GPT-4.

This package provides a detokenizer utilizing the GPT-2 language model, which can be used to reconstruct coherent text from a list of tokens. It decides whether or not to add a space between each token pair, utilizing beam search to maximize the total string probability under the language model.

## Installation
To use this package, you need to have Python installed along with the `transformers` library. You can install `transformers` using pip:

```sh
pip install transformers
```

## Usage
The main class in this package is `GPT2Detokenizer`. Here is a basic usage example:

```python
from detokenizer import GPT2Detokenizer

detokenizer = GPT2Detokenizer()
tokens = ["I", "don ", "'", "t", "know", "."]
print(detokenizer.detokenize(tokens))  # Expected output: "I don't know."
```

## Testing
To run the unit tests, navigate to the directory containing `test_detokenizer.py` in the command line and execute:

```sh
python -m unittest test_detokenizer.py
```
