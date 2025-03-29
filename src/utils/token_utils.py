from typing import Dict, List

tokens_dict = {
    "<object>": "begin object",
    "</object>": "end object",
    "<bbox>": "begin bbox",
    "</bbox>": "end bbox",
    "<class>": "begin class",
    "</class>": "end class",
    "<annotation>": "begin annotation",
    "</annotation>": "end annotation",
}


def generate_coordinate_tokens(num_bins: int, shared_coords: bool = True) -> List[str]:
    """
    Generate special tokens for annotations and coordinates.

    Args:
        num_bins: Number of discrete bins for coordinates
        shared_coords: If True, generates shared x/y coordinate tokens

    Returns:
        List of special tokens
    """
    new_tokens = list(tokens_dict.keys())

    if shared_coords:
        length = len(str(num_bins - 1))
        for i in range(num_bins):
            new_tokens.append(f"<x{i:0{length}d}/>")
            new_tokens.append(f"<y{i:0{length}d}/>")

    return new_tokens


def spatial_position_initialization(
    tokenizer, coordinate_tokens: List[str]
) -> Dict[str, List[int]]:
    """
    Initialize tokens using spatial position concepts already in vocabulary.

    Args:
        tokenizer: Tokenizer to use for encoding
        coordinate_tokens: List of special tokens to initialize
        num_bins: Number of coordinate bins (for potential fallback logic)

    Returns:
        Dictionary mapping tokens to initialization token IDs
    """
    initializers = {}

    # Helper function to get token IDs for a word or phrase
    def get_token_ids(text):
        return tokenizer.encode(text, add_special_tokens=False)

    for token in coordinate_tokens:
        if token in tokens_dict:
            initializers[token] = get_token_ids(tokens_dict[token])
        elif token.startswith("<x"):
            initializers[token] = get_token_ids(f"coordinate x{token[2:-2]}")
        elif token.startswith("<y"):
            initializers[token] = get_token_ids(f"coordinate y{token[2:-2]}")
        else:
            initializers[token] = []

    return initializers


def get_token_initializers(
    tokenizer, coordinate_tokens: List[str]
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
    initializers = spatial_position_initialization(tokenizer, coordinate_tokens)

    return initializers
