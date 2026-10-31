"""Curated English Text Corpora for Transformer Language Modeling.

Provides multiple documented corpora:
1. TinyStories & Fables: Clear sentence structures, narrative progression, and rich vocabulary.
2. AI & Transformer Science: Technical concepts, neural architectures, and language modeling fundamentals.
3. Classic Shakespeare & Poetry: Poetic cadences, rhythmic sequences, and classic phrases.
"""

from typing import Dict

CORPUS_TINY_STORIES = """
Once upon a time, in a bright green forest, there lived a little curious rabbit named Oliver.
Oliver loved to explore the sunny meadows and discover hidden treasures among the tall grass.
Every morning, the golden sun rose over the rolling hills, filling the clear sky with warm amber light.
Oliver would wake up early and say hello to his dear friend, Barnaby the wise old owl.
Barnaby lived in a hollow oak tree at the edge of the crystal river.
The owl knew many fascinating stories about ancient trees, deep rivers, and shimmering stars in the night.
"Good morning, young rabbit," Barnaby said softly with a gentle smile.
"Today is a wonderful day to learn something new about the world around us."
Oliver hopped happily through the field of purple wildflowers and bright red clover.
Along the babbling brook, he met a shy bluebird named Melody who had lost her melody.
Melody sat on a low willow branch, looking down at the clear dancing water.
"Why are you so quiet today, Melody?" Oliver asked with great curiosity.
"I am trying to sing a new song for the forest festival, but I forgot the rhythm," replied the bird.
Oliver thought carefully for a moment and tapped his furry paw against a smooth stone.
Tap, tap, tap went his rhythm against the rock, echoing through the peaceful woods.
Melody listened intently to the steady beat and began to flap her colorful wings.
She opened her small beak and sang a sweet, joyful tune that danced on the gentle breeze.
The song echoed across the sunny meadow, inviting all the woodland creatures to gather together.
Soon, squirrels gathered acorns, deer stepped softly from the shadows, and badgers shared sweet berries.
They all danced in harmony under the tall canopy of emerald leaves.
Oliver felt warm in his heart knowing that friendship and kindness always bring great joy.
When twilight fell over the peaceful forest, the stars began to twinkle like diamonds in the velvet sky.
Barnaby the owl looked down from his hollow oak and whispered to the quiet night.
"Every small act of kindness makes our forest a brighter place for everyone."
Oliver curled up in his warm burrow under the roots of the ancient tree, dreaming of tomorrow's adventures.
The moon watched over the quiet woods, casting silver light across the sleeping flowers.
The wind blew softly through the branches, singing a gentle lullaby to all the creatures of the earth.
Peace and wonder filled the magical realm as another beautiful day came to a quiet close.
"""

CORPUS_AI_SCIENCE = """
A transformer is a deep neural network architecture designed for processing sequential data.
Unlike recurrent neural networks, the transformer processes all tokens simultaneously in parallel.
The fundamental mechanism driving transformers is called scaled dot product self attention.
Self attention allows each token in an input sequence to dynamically attend to every other token.
In self attention, three distinct vector representations are computed: queries, keys, and values.
The query vector represents what the current token is seeking from other tokens in the context.
The key vector serves as a label or address describing the information contained in a token.
The value vector holds the actual semantic content or representation of the token.
Attention weights are computed by taking the dot product of the query and key vectors.
These scores are scaled by the square root of the key dimension to prevent gradient saturation in the softmax.
Softmax normalization converts the raw attention logits into a valid probability distribution over tokens.
For autoregressive language modeling, a causal attention mask is strictly applied to prevent looking ahead.
The causal mask sets all attention scores corresponding to future token positions to negative infinity.
After softmax, future positions receive zero attention weight, preserving the autoregressive property.
Multi head attention projects queries, keys, and values into multiple lower dimensional representation subspaces.
This enables the network to attend to information from different representation positions simultaneously.
Each transformer block combines multi head self attention with a position wise feed forward network.
Residual connections are added around each sub layer, followed by layer normalization.
Positional encodings provide the transformer with essential information about the order of tokens.
Without positional encodings, self attention is completely permutation invariant with respect to sequence order.
Language models are trained using cross entropy loss to minimize next token prediction error.
Perplexity is an exponential measure of cross entropy loss, representing the effective branching factor.
"""

CORPUS_SHAKESPEARE = """
From fairest creatures we desire increase, that thereby beauty's rose might never die.
But as the riper should by time decease, his tender heir might bear his memory.
Thou that art now the world's fresh ornament and only herald to the gaudy spring.
Within thine own bud buriest thy content and, tender churl, mak'st waste in niggarding.
Look in thy glass and tell the face thou viewest, now is the time that face should form another.
Whose fresh repair if now thou not renewest, thou dost beguile the world, unbless some mother.
For where is she so fair whose uneared womb disdains the tillage of thy husbandry?
Or who is he so fond will be the tomb of his self-love, to stop posterity?
Shall I compare thee to a summer's day? Thou art more lovely and more temperate.
Rough winds do shake the darling buds of May, and summer's lease hath all too short a date.
Sometime too hot the eye of heaven shines, and often is his gold complexion dimmed.
And every fair from fair sometime declines, by chance or nature's changing course untrimmed.
But thy eternal summer shall not fade, nor lose possession of that fair thou ow'st.
Nor shall death brag thou wand'rest in his shade, when in eternal lines to time thou grow'st.
So long as men can breathe or eyes can see, so long lives this, and this gives life to thee.
To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune, or to take arms against a sea of troubles
And by opposing end them. To die, to sleep; no more; and by a sleep to say we end
The heart-ache and the thousand natural shocks that flesh is heir to.
"""

# Default curated combined corpus: Rich, coherent, balanced vocabulary
CORPUS_COMBINED = CORPUS_TINY_STORIES.strip() + "\n\n" + CORPUS_AI_SCIENCE.strip() + "\n\n" + CORPUS_SHAKESPEARE.strip()

CORPORA: Dict[str, Dict[str, str]] = {
    "stories": {
        "name": "TinyStories & Woodland Fables",
        "description": "Engaging narrative text with rich syntax, descriptive adjectives, and strong sequential causal structure.",
        "text": CORPUS_TINY_STORIES.strip()
    },
    "science": {
        "name": "Transformer & AI Science",
        "description": "Technical scientific text explaining self-attention, positional encoding, causal masking, and language modeling.",
        "text": CORPUS_AI_SCIENCE.strip()
    },
    "shakespeare": {
        "name": "Shakespeare Sonnets & Soliloquies",
        "description": "Classic English poetic literature featuring famous sonnets, metaphors, and rhythmic phrasing.",
        "text": CORPUS_SHAKESPEARE.strip()
    },
    "combined": {
        "name": "Unified Multi-Domain Corpus (Default)",
        "description": "Comprehensive balanced corpus combining narrative stories, AI technical literature, and poetic text.",
        "text": CORPUS_COMBINED.strip()
    }
}


def get_corpus(corpus_id: str = "combined") -> Dict[str, str]:
    """Retrieve corpus by ID or return default combined corpus."""
    return CORPORA.get(corpus_id, CORPORA["combined"])


def list_corpora() -> Dict[str, Dict[str, str]]:
    """Return dictionary of available corpora with metadata."""
    return {
        cid: {
            "name": meta["name"],
            "description": meta["description"],
            "word_count": len(meta["text"].split()),
            "character_count": len(meta["text"])
        }
        for cid, meta in CORPORA.items()
    }
