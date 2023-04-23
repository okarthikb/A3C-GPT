## A3CGPT

Train GPT with A3C. Because why not (it sucks tbh)...

![a3c](https://user-images.githubusercontent.com/86470305/233813923-c3bf1514-ffe3-446a-a8b3-198cc3b9b1cf.png)

The actor and critic are part of the same LM class (`nn.Linear`s). The critic `V` maps context embeddings to a single scalar value. The actor `π` maps them to a single probability distribution (P(next token)). Choose random action (token) with probability `epsilon` (epsilon-greedy) and with probability `1 - epsilon)` let actor choose token - if token is correct, reward is `1 - p`, if wrong, `-p` (where `p` is predicted probability for that token).
