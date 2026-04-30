def create_tokenizer(name: str, **kwargs):
    """Factory function to create a tokenizer by name."""
    normalized = "wl" if name == "graphs" else name

    if normalized == "wl":
        # Import inside to avoid hard dependency if wlplan is missing
        # and user is using other tokenizers
        try:
            from code.tokenization.wl import WLTokenizer
            return WLTokenizer(iterations=kwargs.get("iterations", 2))
        except ImportError:
            raise ImportError("WLTokenizer requires 'wlplan' package.")
            
    elif normalized == "simhash":
        from code.tokenization.simhash import SimHashTokenizer

        return SimHashTokenizer(
            hash_dim=kwargs.get("hash_dim", 128),
            seed=kwargs.get("seed", 42),
        )
    elif normalized == "shortest_path":
        from code.tokenization.shortest_path import ShortestPathTokenizer

        return ShortestPathTokenizer(
            max_path_length=kwargs.get("max_path_length", 5),
        )
    elif normalized == "graphbpe":
        from code.tokenization.graphbpe import GraphBPETokenizer

        return GraphBPETokenizer(
            vocab_size=kwargs.get("vocab_size", 1000),
            num_iterations=kwargs.get("num_iterations", 100),
        )
    elif normalized == "random":
        from code.tokenization.random import RandomTokenizer

        return RandomTokenizer(
            random_dim=kwargs.get("random_dim", 128),
            seed=kwargs.get("seed", 42),
            normalize=kwargs.get("normalize", True),
        )
    else:
        raise ValueError(f"Unknown tokenizer: {name}")
