import torch


def top_p_filtering(logits, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def word_generating_procedure():
    pass


# Here is how to use this function for top-p sampling
temperature = 1.0
top_p = 0.9

# Get logits with a forward pass in our model (input is pre-defined)
logits = model(input)

# Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
logits = logits[0, -1, :] / temperature
filtered_logits = top_k_top_p_filtering(logits, top_p=top_p)

# Sample from the filtered distribution
probabilities = F.softmax(filtered_logits, dim=-1)
next_token = torch.multinomial(probabilities, 1)
