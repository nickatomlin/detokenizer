from detokenizer import GPT2Detokenizer

def main():
    input_str = input("Enter a string of tokens separated by space: ")
    tokens = input_str.split(' ')
    detokenizer = GPT2Detokenizer()
    detokenized_str = detokenizer.detokenize(tokens)
    print("Detokenized String: ", detokenized_str)
    

if __name__ == "__main__":
    main()