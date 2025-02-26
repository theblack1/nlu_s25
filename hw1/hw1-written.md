# Name: Feifan Liao
# ID:fl2656

# Problem 1c: Extra Credit (Written, 5 Points)
Answer:
the \__getitem__ method assume that "words" is a iterable variable (since: "words: Iterable[str]"), but if we use "embeddings["the"]", the input "words" value is "the", which is a string and will be treated as a iterable variable of ("t", "h", "e") instead of "the" as a whole.   
Therefore, we will not get the embedding result of "the" (unlike embeddings["the", "of"], because the input is a tuple, which allows a proper embedding result of "the" and "of"). What's worse, "h" is not in our "glove_50d.txt" dataset, so the code will lead to a key error of "KeyError: 'h'".

