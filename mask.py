import sys
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import torch
except ImportError:
    torch = None

from PIL import Image, ImageDraw, ImageFont

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

BertMaskedLM = None
BACKEND = None

try:
    if tf is not None:
        try:
            from transformers import TFBertForMaskedLM
            BertMaskedLM = TFBertForMaskedLM
            BACKEND = "tf"
        except ImportError:
            try:
                from transformers import TFAutoModelForMaskedLM
                BertMaskedLM = TFAutoModelForMaskedLM
                BACKEND = "tf"
            except ImportError:
                pass
    if BertMaskedLM is None and torch is not None:
        try:
            from transformers import AutoModelForMaskedLM
            BertMaskedLM = AutoModelForMaskedLM
            BACKEND = "pt"
        except ImportError:
            pass
except Exception:
    BertMaskedLM = None
    BACKEND = None

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = None
GRID_SIZE = 40
PIXELS_PER_WORD = 200
OUTPUT_DIR = Path(__file__).resolve().parent / "attention_maps"


def main():
    if AutoTokenizer is None or BertMaskedLM is None:
        sys.exit("This script requires Transformers and a compatible ML backend.")

    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    mask_token_index = None

    if BACKEND == "tf":
        inputs = tokenizer(text, return_tensors="tf")
        mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
        if mask_token_index is None:
            sys.exit(f"Input must include mask token {tokenizer.mask_token}.")
        model = BertMaskedLM.from_pretrained(MODEL)
        result = model(**inputs, output_attentions=True)
        mask_token_logits = result.logits[0, mask_token_index]
        top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy().tolist()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].numpy().tolist())
    else:
        inputs = tokenizer(text, return_tensors="pt")
        mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
        if mask_token_index is None:
            sys.exit(f"Input must include mask token {tokenizer.mask_token}.")
        model = BertMaskedLM.from_pretrained(MODEL, attn_implementation="eager")
        model.eval()
        with torch.no_grad():
            result = model(**inputs, output_attentions=True)
        mask_token_logits = result.logits[0, mask_token_index]
        top_tokens = torch.topk(mask_token_logits, K).indices.tolist()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    visualize_attentions(tokens, result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """
    input_ids = inputs["input_ids"]

    if hasattr(input_ids, "numpy"):
        input_ids = input_ids.numpy()
    elif hasattr(input_ids, "detach"):
        input_ids = input_ids.detach().cpu().numpy()

    if isinstance(input_ids, (list, tuple)):
        ids = list(input_ids)
    else:
        ids = np.asarray(input_ids).reshape(-1).tolist()

    flat = []
    for item in ids:
        if isinstance(item, (list, tuple, np.ndarray)):
            flat.extend(np.asarray(item).reshape(-1).tolist())
        else:
            flat.append(item)

    for index, token_id in enumerate(flat):
        if token_id == mask_token_id:
            return index
    return None



def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    score = float(attention_score)
    intensity = int(max(0, min(255, round(255 * score))))
    return (intensity, intensity, intensity)



def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).
    """
    for layer_number, layer_attention in enumerate(attentions, start=1):
        if hasattr(layer_attention, "numpy"):
            attention_tensor = layer_attention.numpy()
        elif hasattr(layer_attention, "detach"):
            attention_tensor = layer_attention.detach().cpu().numpy()
        else:
            attention_tensor = np.asarray(layer_attention)

        if attention_tensor.ndim == 4:
            heads = attention_tensor[0]
        elif attention_tensor.ndim == 3:
            heads = attention_tensor
        elif attention_tensor.ndim == 2:
            heads = attention_tensor[np.newaxis, ...]
        else:
            continue

        for head_number, head_attention in enumerate(heads, start=1):
            generate_diagram(layer_number, head_number, tokens, head_attention)


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """
    global FONT
    if FONT is None:
        try:
            FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
        except OSError:
            FONT = ImageFont.load_default()

    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    OUTPUT_DIR.mkdir(exist_ok=True)
    img.save(OUTPUT_DIR / f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()
