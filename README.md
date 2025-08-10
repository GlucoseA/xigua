# xigua

This repository provides a minimal demo of an input method powered by a Transformer model. The demo uses the pre-trained GPT-2 model to predict the next word based on context. As you type characters, the script updates candidate suggestions.

## Usage

Install dependencies (requires internet access):

```bash
pip install torch transformers
```

Run the demo:

```bash
python demo.py
```

You will be prompted for an initial context. Then, type characters one by one. After each character, the script displays suggestions for completing the current word. Press Enter to accept the word and continue, or type `/quit` to exit.
