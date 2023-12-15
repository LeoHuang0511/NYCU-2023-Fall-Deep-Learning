def main():
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow as tf
    tf.test.gpu_device_name()

    train_data_path = 'shakespeare_train.txt'
    val_data_path = 'shakespeare_valid.txt'

    with open(train_data_path, 'r', encoding='UTF-8') as f:
        train_text = f.read()

    with open(val_data_path, 'r', encoding='UTF-8') as f:
        valid_text = f.read()

    vocab_to_int = tf.keras.layers.StringLookup(
        output_mode='int',
        vocabulary=list(set(train_text)),
    )

    int_to_vocab = tf.keras.layers.StringLookup(
        output_mode='int',
        vocabulary=vocab_to_int.get_vocabulary(),
        invert=True,
    )

    train_data = tf.strings.unicode_split(train_text, 'UTF-8')
    train_data = vocab_to_int(train_data)

    valid_data = tf.strings.unicode_split(valid_text, 'UTF-8')
    valid_data = vocab_to_int(valid_data)





    SEQ_LENGTH = 20
    BATCI_SIZE = 1024


    def create_ds(train_data, valid_data, seq_length, batch_size):
        
        train_ds = tf.keras.utils.timeseries_dataset_from_array(
            train_data[:-1],
            train_data[seq_length:],
            seq_length,
            batch_size=batch_size,
            shuffle=True,
        ).cache().prefetch(tf.data.AUTOTUNE)

        valid_ds = tf.keras.utils.timeseries_dataset_from_array(
            valid_data[:-1],
            valid_data[seq_length:],
            seq_length,
            batch_size=batch_size,
        ).cache().prefetch(tf.data.AUTOTUNE)
        return train_ds, valid_ds


    train_ds, valid_ds = create_ds(train_data, valid_data, SEQ_LENGTH, BATCI_SIZE)





    HIDDEN_SIZE = 128
    RNN_TYPE = 'lstm'

    if RNN_TYPE == 'rnn':
        rnn_layer = tf.keras.layers.SimpleRNN(HIDDEN_SIZE)
    else:  
        rnn_layer = tf.keras.layers.LSTM(HIDDEN_SIZE)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_to_int.vocabulary_size(), 32),
        rnn_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(int_to_vocab.vocabulary_size()),
    ])

    model.summary()





    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')


    @tf.function
    def train_step(model, x, y, loss_fn, optimizer):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y, y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss(loss)
        train_accuracy(y, y_pred)


    @tf.function
    def test_step(model, x, y, loss_fn):
        y_pred = model(x, training=False)
        loss = loss_fn(y, y_pred)

        test_loss(loss)
        test_accuracy(y, y_pred)


    @tf.function
    def predict(
        input_seq: tf.Tensor,
        size: int = 1,
        model: tf.keras.Model = model,
    ):
        for _ in range(size):
            pred = model(input_seq)
            pred = tf.random.categorical(pred, num_samples=1)
            input_seq = tf.concat([input_seq, pred], axis=-1)
        return input_seq





    sample_data = next(iter(train_ds))[0][:4]
    sample_text = tf.strings.reduce_join(int_to_vocab(sample_data), axis=-1)
    print('Sample text:', sample_text.numpy())

    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    log_dir = os.path.join('logs', f'{RNN_TYPE}_{HIDDEN_SIZE}_seq_{SEQ_LENGTH}')
    train_log_dir = os.path.join(log_dir, 'train')
    test_log_dir = os.path.join(log_dir, 'test')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)


    with train_summary_writer.as_default():
        tf.summary.text('sample_text', sample_text, step=0)





    from tqdm import tqdm

    EPOCHS = 30

    for epoch in range(1, EPOCHS + 1):
        with tqdm(
                train_ds,
                total=train_ds.cardinality().numpy(),
                desc=f'Epoch {epoch} / {EPOCHS} @ train',
                dynamic_ncols=True,
        ) as pbar:
            for x, y in pbar:
                train_step(model, x, y, loss_fn, optimizer)

                pbar.set_postfix_str(
                    f'loss: {train_loss.result():.6f}, accuracy: {train_accuracy.result():.6f}'
                )

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

            if epoch % (EPOCHS // 5) == 0:
                pred = predict(sample_data, 100)
                pred = int_to_vocab(pred)
                pred = tf.strings.reduce_join(pred, axis=-1)
                tf.summary.text('breakpoint_text', pred, step=epoch)

        with tqdm(
                valid_ds,
                total=valid_ds.cardinality().numpy(),
                desc=f'Epoch {epoch} / {EPOCHS} @ test',
                dynamic_ncols=True,
        ) as pbar:
            for x, y in pbar:
                test_step(model, x, y, loss_fn)

                pbar.set_postfix_str(
                    f'loss: {test_loss.result():.6f}, accuracy: {test_accuracy.result():.6f}'
                )

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)





    def prime_text(text, prime_size):
        prime_seq = tf.strings.unicode_split(text, 'UTF-8')
        prime_seq = vocab_to_int(prime_seq)
        prime_seq = tf.expand_dims(prime_seq, axis=0)

        pred = predict(prime_seq, prime_size)

        pred = int_to_vocab(pred)
        pred = tf.strings.reduce_join(pred)
        return pred.numpy().decode('UTF-8')





    prime_result = prime_text('JULIET', 500)

    print(prime_result)


if __name__ == '__main__':
    main()
