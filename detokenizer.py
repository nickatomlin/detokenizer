import itertools
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Detokenizer:
    def __init__(self, model_name='gpt2'):  # corrected the model_name to 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
    
    def detokenize(self, tokens, beam_width=10):
        if type(tokens) == str:
            tokens = tokens.split()
        beam_candidates = [([], 0)]  # Each candidate is represented as ([spaces], log_prob)
        
        # Loop over each pair of tokens and decide whether to insert space or not
        for i in range(len(tokens)-1):
            new_candidates = []
            for candidate, log_prob in beam_candidates:
                for is_space in [True, False]:
                    new_candidate = candidate + [is_space]
                    candidate_str = self.construct_string(tokens, new_candidate)
                    token_ids = self.tokenizer.encode(candidate_str, return_tensors='pt')
                    output = self.model(token_ids)
                    # Accumulate the log probability of the sequence
                    log_prob_seq = output.logits[0, :-1].log_softmax(dim=-1).gather(dim=-1, index=token_ids[0, 1:].unsqueeze(-1)).sum().item()
                    new_log_prob = log_prob + log_prob_seq
                    new_candidates.append((new_candidate, new_log_prob))
            
            # Keep only the top-k candidates
            beam_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        best_candidate = beam_candidates[0][0]
        return self.construct_string(tokens, best_candidate)
    
    def construct_string(self, tokens, spaces):
        result = [tokens[0]]
        for token, is_space in zip(tokens[1:], spaces):
            if is_space:
                result.append(' ')
            result.append(token)
        return ''.join(result)

# Example usage:
if __name__ == "__main__":
    detokenizer = GPT2Detokenizer()
    tokens = ["I", "can", "'", "t", "figure", "this", "out", "."]
    print(detokenizer.detokenize(tokens))  # Expected output: "I love Python programming."
