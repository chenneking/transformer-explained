# An implementation of the (encoder-only) transformer architecture from the paper Attention Is All You Need (Vaswani et al.)
# Includes an extensive amount of comments to explain every single step in detail.

# Import
import torch
import torch.nn as nn
import math

"""
Defines the logic used to add a positional "component" to the input embeddings.

We require this positional information because otherwise, due to the parallel nature of transformers, we would lose all positional information.
For example, "the cat sat on the man" vs. "the man sat on the cat", have two very different meanings (poor cat), and without positional encoding, 
the transformer wouldn't really be able to differentiate between the two, that's why this is important!
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len:int) -> None:
        """
        :param d_model: The embedding dimension used in the model.
        :param max_seq_len: The maximum length of the output sequence as measured in no. of tokens.
        """
        super(PositionalEncoding, self).__init__()

        # We're going to define a matrix that we end up adding to the input matrix (a concatenated set of embedding vectors)
        # Also, we define this matrix to cover the worst-case (when we reach the maximum sequence length) and when it's used, we just use the first columns

        # Start off with a matrix filled with zeros (max_seq_len, d_model)
        pos_encoding: torch.Tensor = torch.zeros(max_seq_len, d_model)

        # Define indices for each token in the sequence (max_seq_len,)
        pos: torch.Tensor = torch.arange(0, max_seq_len).float()
        # Turn that vector from 1D (max_seq_len,) into 2D (max_seq_len, 1) to match dimensions for the following steps
        pos = pos.reshape(max_seq_len, 1)

        # As defined in the paper we use the following values at each position:
        # PE(pos, 2i) = sin(pos/10000^{2i/d_model})
        # PE(pos, 2i+1) = cos(pos/10000^{2i/d_model})
        # Which is equivalent to:
        # PE(pos, 2i) = sin(pos * exp(-log(10000) * 2i / d_model))
        # PE(pos, 2i+1) = cos(pos * exp(-log(10000) * 2i / d_model))
        # Remember: pos is the index of the token in the sequence, and i is the index of the embedding dimension!

        # Here we implement the equation just as described in the comment above.
        # Because we have use 2i, we can use a tensor that starts at 0 and increments by 2 each step: [0, 2, 4, ...]
        factor_term: torch.Tensor = torch.exp(-1 * math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)

        # Now lets set these values in our matrix of zeros we defined above (pos_encoding)
        # We apply sin to every row and every even column index (each row represents 1 embedded token)
        pos_encoding[:, 0::2] = torch.sin(pos * factor_term)
        # We apply cos to every row and every odd column index
        pos_encoding[:, 1::2] = torch.cos(pos * factor_term)

        # Because we only want to compute this positional encoding once, we want to register a buffer with it. This is
        # because the pos_encoding matrix doesn't change at all, we just apply different number of rows to an input tensor x.
        self.register_buffer("pos_encoding", pos_encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the positional encoding to the input x.
        :param x: Input sequence (a matrix of concatenated embeddings).
        :return: Input sequence with positional encoding.
        """
        # Depending on how long the input sequence in x is, we have to apply a subset of rows of our positional encoding matrix.
        # Therefore, we cut the matrix off at row index (equiv to sequence token number) x.size(1)
        return x + self.pos_encoding[:, :x.size(1), :]


"""
Defines the logic used to do self-attention with multiple heads.

We use multiple heads so that we can capture different aspects of relationships in the sequence, and we do this in parallel
to make it more efficient.

Now you might ask: Why isn't it enough to keep every encoder layer as a single "head" and just rely on the multiple 
sequential layers to capture these relationships? 
The two key reasons are: multiple heads make this process more efficient and more expressive: The individual heads are 
computed in parallel, and a single head can only capture one relationship at a time. Since natural language contains many
complex relationships (e.g. word meanings based on context) we want to capture as many of these as possible!
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int) -> None:
        super(MultiHeadAttention, self).__init__()

        self.num_heads: int = num_heads
        # We need to ensure that the embedding dimension is a multiple of the number of heads!
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Because each head gets a chunk of the W_q, W_k, and W_v matrices (the matrices that transform the input to Q,
        # K, and V respectively), we split these chunks up equally by the number of heads.
        # Although the paper defines d_k and d_v seperately, we assume d_k = d_v.
        self.d_k: int = d_model // num_heads

        # Although these are just weight matrices, we want to be able to learn them. That's why we use linear layers, as
        # we can backpropagate through these layers and ensure these matrices are actually "learned".
        # But because we treat these as matrices, we don't need a bias vector!
        self.W_q = nn.Linear(d_model, d_model, bias=False) # (d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model, bias=False) # (d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model, bias=False) # (d_model, d_model)

        # Because we split the 3 weight matrices up, we have to aggregate the final result to get a matrix of shape
        # (d_model, d_model) as output
        # Although we aggregate the attention results of all heads, we always know that d_k * num_heads = d_model
        # (see the assert statement above), thus, we can just use (d_model, d_model) for the W_o matrix.
        self.W_o = nn.Linear(d_model, d_model, bias=False) # (d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (split) input x into multiple heads.
        :param x: input tensor
        :return: reshaped tensor
        """
        batch_size, seq_len, d_model = x.size()
        # Here we split the last dimension of the tensor d_model (the embedding dimension) into two dimensions:
        # num_heads and d_k (where d_model = num_heads * d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k) # (batch_size, seq_len, num_heads, d_k)
        # Lastly, we swap the order of the dimensions.
        # Why you ask? Because the attention operation has to operate on each head independently, so that the respective
        # sequence "given" to each head can be processed in parallel!
        return x.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, d_k)

    def compute_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the attention scores between Q, K and V.
        :param Q: The Q matrix.
        :param K: The K matrix.
        :param V: The V matrix.
        :return: The attention score matrix.
        """
        # When transposing K we need to switch the correct dimensions: seq_len and d_k to make sure the dot product is
        # multiplying the right dimensions with each other.
        scores: torch.Tensor = Q @ K.transpose(-1, -2) # (batch_size, num_heads, seq_len, seq_len)

        # We divide by the square root of d_k.
        # Now you might ask: Why and why specifically normalize by sqrt(d_k)?
        # This term helps stabilize the variance of each element in the dot product of Q and K^T. Under the assumption
        # that every element in Q and K is drawn from a standard normal distribution, we can show that with this normalization
        # term makes the variance of every element 1, after we divide by the sqrt of d_k.
        scaled_scores: torch.Tensor = scores/math.sqrt(self.d_k) # (batch_size, num_heads, seq_len, seq_len)

        # If we have a mask, e.g. to ensure causality (aka make sure we can only look at past tokens and not future ones)
        # we can mask out attention scores using the following:
        # This assumes the mask has 1s for positions to keep and 0s for positions to set to -inf
        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))

        # Here we do row-wise softmax. Remember: each row represents a token, so we want the attention weights we just
        # computed for each key to add up to 1.
        normalized_scores = torch.softmax(scaled_scores, dim=-1) # (batch_size, num_heads, seq_len, seq_len)

        # Finally we get the dot product with the value matrix.
        attention: torch.Tensor = normalized_scores @ V # (batch_size, num_heads, seq_len, d_k)
        return attention

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combines the attention scores from all heads through concatenation.
        :param x: Tensor that contains the attention scores from the different heads.
        :return: Tensor that with merged attention scores.
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        # here we switch the order of dimensions from (batch_size, num_heads, seq_len, d_k) to
        # (batch_size, seq_len, num_heads, d_k) to prepare to squish the last two dimensions (through concatenation)
        x = x.permute(0, 2, 1, 3)
        # We need to move the tensor to a contigous chunk of memory to make sure we can use view() on it properly
        x = x.contiguous()
        # Finally, we squish the last two dimensions num_heads, d_k to d_model
        return x.view(batch_size, seq_len, self.d_model) # (batch_size, num_heads, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute a forward pass using multi-head attention.
        :param x: input tensor
        :param mask: Optional mask tensor (e.g to enforce causality in the encoder)
        :return:
        """

        # Now let us finally actually compute the Q, K, and V matrices and immediately split these up into multiple heads.

        # Now you may ask: Why are these called Q, K, and V? These are abbreviations for Query, Key, and Value matrices.
        # These terms come from the field of information retrieval. Let's consider these terms in the context of e.g.
        # SQL Database: What you're looking for is the query (select id from users), what you're matching against is the
        # key (the id is the primary key of the table users), and the rows you return is the value(s).

        # Now you might also ask: But these transformations are all applied to the same input x, how does that make sense?
        # Here's a great analogy I was taught: Think of this in the context of online dating.
        # Q is your preferences for age, height, etc. (this is what you're looking for)
        # K is your own profile including all your pictures and bio (this is what you want others to match against)
        # V is how you actually show up for the date (your true value)
        Q: torch.Tensor = self.split_heads(self.W_q(x)) # (batch_size, num_heads, seq_len, d_k)
        K: torch.Tensor = self.split_heads(self.W_k(x)) # (batch_size, num_heads, seq_len, d_k)
        V: torch.Tensor = self.split_heads(self.W_v(x)) # (batch_size, num_heads, seq_len, d_k)

        # Here we actually compute the attention score matrix and apply a mask if provided
        x = self.compute_attention(Q, K, V, mask) # (batch_size, num_heads, seq_len, d_k)

        # Because we have multiple heads, we need to combine their individual results through concatenation
        x = self.combine_heads(x) # (batch_size, num_heads, d_model)

        # Finally we project the concatenated outputs down to the output dimension
        x = self.W_o(x) # (batch_size, num_heads, d_model)
        return x


"""
Defines the logic used in the Feed-Forward Networks used in every encoder block.
"""
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int):
        """
        Defines the feed-forward network architecture.
        :param d_model: Model dimension (also embedding dimension)
        :param d_ff: Feed forward network hidden dimension (the dimension of the hidden state)
        """
        super(FeedForward, self).__init__()

        # As described in the paper, we use two linear layers with a ReLU activation in between.
        self.relu = nn.ReLU()
        # The first layer projects from the embedding dimension d_model to the feed-forward hidden layer dimension d_ff.
        self.fc1 = nn.Linear(d_model, d_ff)
        # The second layer projects back to the embedding dimension d_model.
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # We apply the first linear layer, then the ReLU activation, and lastly the second linear layer.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

"""
Combines the previously defined components into a unified encoder layer
"""
class EncoderLayer(nn.Module):
    def __init__(self, d_model:int, d_ff:int, num_heads:int, dropout:float):
        """
        Defines the encoder layer architecture.
        :param d_model: Model dimension (also embedding dimension)
        :param d_ff: Feed forward network hidden dimension (the dimension of the hidden state)
        :param num_heads: Number of attention heads used in multi-head attention.
        :param dropout: Dropout probability.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn_masked = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        # Define a residual value that's added back
        residual: torch.Tensor = x

        # First we apply masked multi-head attention to the input tensor x
        x = self.self_attn_masked(x, mask)
        # Next, we do a dropout, add the residual connection and batch norm the tensor
        x = self.dropout(x)
        x = x + residual
        x = self.norm1(x)

        # Reset the residual so that it can be used later on again
        residual = x

        # Because we are using an encoder-only model, we don't have cross-attention and move right to the feed forward network
        # Apply the feedforward network
        x = self.feed_forward(x)

        # Again, we do a dropout, add the residual connection and batch norm the tensor
        x = self.dropout(x)
        x = x + residual
        x = self.norm2(x)
        return x

"""
Combines all the previous classes to build the final Transformer model.
"""
class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, num_layers: int, num_heads: int, d_ff: int, dropout: float):
        """
        :param vocab_size: Size of the vocabulary.
        :param d_model: The embedding dimension used in the model.
        :param max_seq_len: The maximum length of the output sequence as measured in no. of tokens.
        :param num_layers: The number of encoder layers.
        :param num_heads: The number of attention heads we use in each encoder layer (multi-head attention).
        :param d_ff: Hidden dimension of the feed-forward network in each encoder layer.
        :param dropout: Dropout probability of the attention heads.
        """
        super(Transformer, self).__init__()

        # Include the positional encoding to use on the input tensor
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # Define the sequence of num_layers encoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )

        # We want to learn a mapping from the output representation retrieved from the final encoder layer to the available
        # next tokens available (our entire vocabulary)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        # First, we apply the positional encoding to ensure the transformer can actually understand positional relationships
        x = self.positional_encoding(x) # (batch_size, seq_len, d_model)

        # Next we pass the input through the set of encoder layers
        for encoder in self.encoder_layers:
            x = encoder(x, mask) # (batch_size, seq_len, d_model)

        # We pass the output from the final encoder layer to the final linear layer to get logits for the entire vocabulary
        logits: torch.Tensor = self.fc(x) # (batch_size, seq_len, vocab_size)
        # And instead of returning the softmax over these logits, we return the raw logits as they're used to calculate
        # losses like the CrossEntropyLoss
        return logits
