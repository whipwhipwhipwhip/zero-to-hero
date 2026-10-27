## Zero-to-hero

This is a repo that is mostly copied over from a private repo I had where I was learning RL/LLM stuff. It got too sprawling so I broke it off into sensible parts. This one is notes from where I followed Karpathy's zero-to-hero lectures.

You can find a playlist of those lectures [here](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).


Between the notebooks and src/ folder, I divided the material from his lectures into five parts:
1. Micrograd ([notebooks](notebooks/01_micrograd), [src](src/zero_to_hero/part_01_micrograd)) - material from the youtube video "The spelled-out intro to neural networks and backpropagation: building micrograd". It also has exercises from that lecture, and my extension of the micrograd engine to vector-valued inputs.
2. Makemore ([notebooks](notebooks/02_makemore), [src](src/zero_to_hero/part_02_makemore)) - Material from the five youtube videos "Building makemore" (parts 1,2,3,5), including some exercises.
3. Backpropagation ([src](src/zero_to_hero/part_03_backprop)) - Material from "Building makemore: Part 4", which deals with manual backpropagation. The src/ files just have some basic manual code.
4. GPT ([notebooks](notebooks/04_gpt), [src](src/zero_to_hero/part_04_gpt)) - Material from "Let's build GPT: from scratch, in code, spelled out." This is a more "basic" GPT architecture, the GPT-2 reproduction (along with other simple GPT experiments I did) is in XXX.
5. Tokenizers ([notebooks](notebooks/05_tokenizers)) - Material from "Let's build the GPT tokenizer". Note that there are only some demonstration notebooks. All the source code exists in my [nanotok](https://github.com/whipwhipwhipwhip/nanotok) repo.
