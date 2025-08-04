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


def generate_coordinate_tokens(num_bins: int, num_query_tokens: int = 0) -> List[str]:
    """
    Generate special tokens for annotations and coordinates.

    Args:
        num_bins: Number of discrete bins for coordinates
        num_query_tokens: Number of query tokens to generate

    Returns:
        List of special tokens
    """
    new_tokens = list(tokens_dict.keys())

    # Generate coordinate tokens
    length = len(str(num_bins - 1))  # Calculate string length for formatting
    for i in range(num_bins):
        new_tokens.append(f"<x{i:0{length}d}/>")
        new_tokens.append(f"<y{i:0{length}d}/>")

    length = len(str(num_query_tokens - 1))
    for i in range(num_query_tokens):
        new_tokens.append(f"<query{i:0{length}d}/>")

    return new_tokens


def get_token_initializers(
    tokenizer, coordinate_tokens: List[str]
) -> Dict[str, List[int]]:
    """
    Create initializers for added tokens, mapping them to semantically
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

    for token in coordinate_tokens:
        if token in tokens_dict:
            initializers[token] = get_token_ids(tokens_dict[token])
        elif token.startswith("<x"):
            initializers[token] = get_token_ids(f"coordinate x{token[2:-2]}")
        elif token.startswith("<y"):
            initializers[token] = get_token_ids(f"coordinate y{token[2:-2]}")
        elif token.startswith("<query"):
            initializers[token] = get_token_ids(f"coordinate query{token[6:-2]}") # TODO: rename to object 1, object 2
        else:
            initializers[token] = []

    return initializers
