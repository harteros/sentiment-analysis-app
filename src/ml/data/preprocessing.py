import re


# Preprocess function


def preprocess(x):
    # Make text lower case
    x = x.lower()

    # Remove tags of other people
    x = re.sub(r"@\w*", " ", x)

    # Remove special characters
    x = re.sub(r"#|^\*|\*$|&quot;|&gt;|&lt;|&lt;3", " ", x)
    x = x.replace("&amp;", " and ")

    # Remove links
    x = re.sub(r"ht+p+s?://\S*", " ", x)

    # Remove non-ascii
    x = re.sub(r"[^\x00-\x7F]", " ", x)

    # Remove time
    x = re.sub(r"((a|p).?m)?\s?(\d+(:|.)?\d+)\s?((a|p).?m)?", " ", x)

    # Remove brackets if left after removing time
    x = re.sub(r"\(\)|\[\]|\{\}", " ", x)

    # For words we want to keep at least two occurences of
    #  each word(e.g not change good to god)
    x = re.sub(r"([a-z])\1+", r"\1\1", x)

    # Remove any string that starts with number
    x = re.sub(r"\d[\w]*", " ", x)

    # Remove all special characters left
    x = re.sub(r"[^a-zA-Z0-9 ]", "", x)

    # Remove single letters that left except i and a
    x = re.sub(r"\s[b-gj-z]\s", " ", x)

    # Remove multiple space chars
    x = " ".join(x.split()).strip()

    return x
