from typing import Dict, List


def generate_coordinate_tokens(num_bins: int, shared_coords: bool = True) -> List[str]:
    """
    Generate quantized coordinate tokens for bounding box representation.
    Can generate either shared tokens across all coordinates or separate tokens for x0,y0,x1,y1.
    shared_coords=True: num_bins
    shared_coords=False: num_bins * 4

    Args:
        num_bins: Number of quantization bins per dimension
        shared_coords: If True, creates shared tokens for all coordinates (num_bins total)
                     If False, creates separate tokens for x0,y0,x1,y1 (num_bins * 4 total)

    Returns:
        List of special tokens for coordinates
    """
    coordinate_tokens = []

    if shared_coords:
        # Generate shared tokens for all coordinates
        for i in range(num_bins):
            # pct = i * 100 // (num_bins - 1) if num_bins > 1 else 0
            #token = f"<coord_{pct}>"
            token = f"<coord_{i}>"
            coordinate_tokens.append(token)
    else:
        # Generate separate tokens for each coordinate
        for coord in ["x0", "y0", "x1", "y1"]:
            for i in range(num_bins):
                #pct = i * 100 // (num_bins - 1) if num_bins > 1 else 0
                #token = f"<{coord}_{pct}>"
                token = f"<{coord}_{i}>"
                coordinate_tokens.append(token)

    return coordinate_tokens


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
    for token in coordinate_tokens:
        initializers[token] = []

    return initializers


if __name__ == "__main__":
    # Generate coordinate tokens
    coordinate_tokens = generate_coordinate_tokens()
    print(f"Generated {len(coordinate_tokens)} coordinate tokens")

    # Load a tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("lmms-lab/llava-onevision-qwen2-0.5b-si")

    # Get initializers for coordinate tokens
    initializers = get_token_initializers(tokenizer, coordinate_tokens)
    print(f"Generated initializers for {len(initializers)} tokens")

    # Print examples
    for i, token in enumerate(list(initializers.keys())[:]):
        print(f"{token}: {tokenizer.decode(initializers[token])}")
