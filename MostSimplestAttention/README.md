Reference: https://github.com/philipperemy/keras-attention-mechanism

Application Purpose
    Given sequence, find the hidden rule from the answer
    For example, given x == [1, 2, 3, 4, 5, 5, 4, 1, 3, 2], y == 5
    What is the rule? add two numbers which are the number after 1
    In this case, 2 + 3 = 5

Of course, fully connected layer cannot logically process arithmetic calculation
There will be a small validation loss

Network Architecture

1) Find the correlation of sequence 
 - Each output will depends on the past history --> LSTM

2) From the correlation vector, calculate attention -> Self Attention
 - We want to highest attention score at the numbers where is after '1'

3) Add --> Dense (output will be 1)


Input = [batch size, seq length, 1]
LSTM = in: [batch size, seq length, 1] out: [batch size, seq length, 100] 100 is encoded vector by LSTM
Attention = in: [batch size, seq length, 100] out: [batch size, 128] 128 is kind of latent vector for calculation
Dense = in: [batch size, 128] out: [batch size, 1] 1 is the calculation output

Self Attention (internal dim is 100)
Query
 - Input * Dense(internal dim) => (batch size, seq length, internal dim)
Key
 - Input * Dense(internal dim) => (batch size, seq length, internal dim)
Value
 - Input * Dense(internal dim) => (batch size, seq length, internal dim)

Attention = softmax(Query * reverse(Key)) => (batch size, seq length, seq length)
Attention * Value = (batch, seq length, seq length) x (batch, seq, internal dim)
( Td x Tk ) x (Tk x dim) = Td x dim
After that Max along axis == 1 => [batch, seq]
Dense(128) => [batch, 128]

Input can be represented on [1 x internal dim]
value represents the influence of 2 dims (positive and negative)
We can represent Query and Key with [1 x internal dim] for each sequence
For each dim, multiply (dot product) 
[1 x internal dim] x [1 x internal dim] -> [1 x 1]... Note that each dim is the same dimension
It means it calculates similiarty (dot product)
Not to normalize? yes, because it is relative similarity (and after that, softmax will solve this problem)
After softmax, there will be the highest similarity on Tk(encoder output) space for each Td (decoder output)

