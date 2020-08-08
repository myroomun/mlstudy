from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate, Softmax
import tensorflow as tf


def attention_3d_block(hidden_states):
    """
    Many-to-one attention mechanism for Keras.
    @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    @return: 2D tensor with shape (batch_size, 128)
    @author: felixhao28.
    """
    if False:
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector

    """
    Many-to-one attention mechanism for Keras. (modified version)
    @author: ysmoon 
    """
    hidden_size = int(hidden_states.shape[2])
    query = Dense(hidden_size, use_bias = False, name = "query")(hidden_states)
    key = Dense(hidden_size, use_bias = False, name = "key")(hidden_states)
    score = dot([query, key], [2, 2])  # [batch, seq, seq]
    attention_weights = Activation('softmax', name = 'attention_weight')(score)
    value = Dense(hidden_size, use_bias = False, name = "value")(hidden_states)
    context_vector = dot([attention_weights, value], [2, 1])
    context = tf.keras.backend.max(context_vector, axis = 2)
    attention_vector = Dense(128, use_bias = False, activation = 'tanh', name = 'attention_vector')(context)
    return attention_vector