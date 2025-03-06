from typing import Dict, List


def generate_coordinate_tokens(num_bins: int, shared_coords: bool = True) -> List[str]:
    new_tokens = []

    new_tokens = ["<object>", "</object>", "<bbox>", "</bbox>", "<class>", "</class>", "<annotation>", "</annotation>"]
    
    return new_tokens

def spatial_position_initialization(tokenizer, coordinate_tokens, num_bins):
    """Initialize tokens using spatial position concepts already in vocabulary."""
    initializers = {}
    
    # Helper function to get token IDs for a word or phrase
    def get_token_ids(text):
        return tokenizer.encode(text, add_special_tokens=False)
    

    
    for token in coordinate_tokens:
        if token == "<annotation>":
            initializers[token] = get_token_ids("<annotation")
        elif token == "</annotation>":
            initializers[token] = get_token_ids("</annotation")
        elif token == "<object>":
            initializers[token] = get_token_ids("<object")
        elif token == "</object>":
            initializers[token] = get_token_ids("</object")
        elif token == "<bbox>":
            initializers[token] = get_token_ids("<bbox")
        elif token == "</bbox>":
            initializers[token] = get_token_ids("</bbox>")
        elif token == "<class>":
            initializers[token] = get_token_ids("<class")
        elif token == "</class>":
            initializers[token] = get_token_ids("</class")
        else:
            initializers[token] = []

    
    return initializers

def get_token_initializers(
    tokenizer, coordinate_tokens: List[str], num_bin: int
) -> Dict[str, List[int]]:
    """
    Create initializers for coordinate tokens, mapping them to semantically
    similar tokens in the existing vocabulary.

    Args:
        tokenizer: The tokenizer to use for encoding
        coordinate_tokens: List of special tokens to initialize

    Returns:
        Dictionary mapping token to list of token IDs for initialization
    """
    initializers = {}

    # Helper function to get token IDs for a word or phrase
    def get_token_ids(text):
        return tokenizer.encode(text, add_special_tokens=False)

    # for token in coordinate_tokens:
    #     # Extract the token type and value
    #     if token.startswith("<x") or token.startswith("<y"):
    #         coord_type = token[1:3]  # x0, y0, x1, y1
    #         if "_" in token:
    #             value = token.split("_")[1].rstrip(">")
    #             # Initialize with semantically similar tokens
    #             if coord_type == "x0":
    #                 initializers[token] = get_token_ids(f"left {value}%")
    #             elif coord_type == "y0":
    #                 initializers[token] = get_token_ids(f"top {value}%")
    #             elif coord_type == "x1":
    #                 initializers[token] = get_token_ids(f"right {value}%")
    #             elif coord_type == "y1":
    #                 initializers[token] = get_token_ids(f"bottom {value}%")

    # TODO:more sophisticated initializers

    # Init with empty list for now
    # for token in coordinate_tokens:
    #     initializers[token] = []

    # Init with spatial position concepts
    initializers = spatial_position_initialization(tokenizer, coordinate_tokens, num_bin)

    return initializers


if __name__ == "__main__":
    # Generate coordinate tokens
    coordinate_tokens = generate_coordinate_tokens(100)
    print(f"Generated {len(coordinate_tokens)} coordinate tokens")

    # Load a tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("lmms-lab/llava-onevision-qwen2-0.5b-si")

    # Get initializers for coordinate tokens
    initializers = get_token_initializers(tokenizer, coordinate_tokens, 100)
    print(f"Generated initializers for {len(initializers)} tokens")

    # Print examples
    for i, token in enumerate(list(initializers.keys())[:]):
        print(f"{token}: {tokenizer.decode(initializers[token])}")
