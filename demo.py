import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 tokenizer and model
# Using small gpt2 model for demonstration

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()


def get_suggestions(context: str, prefix: str, k: int = 5):
    """Return up to k suggestions for the next word starting with prefix."""
    # Encode the context
    input_ids = tokenizer.encode(context, return_tensors='pt')
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]
    # Take a larger top_k to improve chances of matching prefix
    topk = torch.topk(logits, k=50)
    suggestions = []
    for token_id in topk.indices.tolist():
        token_str = tokenizer.decode([token_id]).strip()
        if token_str.startswith(prefix):
            suggestions.append(token_str)
        if len(suggestions) >= k:
            break
    return suggestions


def interactive_loop():
    print("Simple input method demo using GPT-2. Type /quit to exit.")
    context = input("Initial context: ")
    prefix = ""
    while True:
        ch = input("Type next character (Enter to commit word): ")
        if ch == "/quit":
            break
        if ch == "":
            context += prefix + " "
            prefix = ""
            print(f"\nUpdated context: {context}\n")
            continue
        prefix += ch
        suggestions = get_suggestions(context, prefix)
        print("Suggestions:", suggestions)


if __name__ == "__main__":
    interactive_loop()
