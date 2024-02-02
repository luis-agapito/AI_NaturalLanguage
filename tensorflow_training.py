import os
import pickle
import random
import string
import tensorflow as tf 
import torchtext
import transformers

import notebooks.natural_language_utils as utils
transformers.logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def buildModelAndTrain(hyperparams, inputs_and_labels):
    print("Training: name={:}, batch_size={:}, num_epochs={:}, learning rate={:}".format(
        hyperparams['name'],
        hyperparams['batch_size'],
        hyperparams['num_epochs'],
        hyperparams['learning_rate']
    ))
        
    num_epochs = hyperparams['num_epochs']
    num_batches_print = hyperparams['num_batches_print']
    learning_rate = hyperparams['learning_rate']

    model = transformers.TFDistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    crossentropy_loss_start = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    crossentropy_loss_end = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    losses = []
    for epoch_id in range(num_epochs):
        print(f"Avg. loss for epoch#{epoch_id:02d}")
        for batch_index, (batch_inputs, batch_labels) in enumerate(inputs_and_labels):
            with tf.GradientTape() as tape:
                outputs = model(batch_inputs)
                loss_start = crossentropy_loss_start(batch_labels['start_positions'], 
                                                     outputs.start_logits)
                loss_end = crossentropy_loss_end(batch_labels['end_positions'],
                                                 outputs.end_logits)
                loss_avg =  (loss_start + loss_end) / 2.0

            grads = tape.gradient(loss_avg, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
            losses.append(loss_avg)
    
            if batch_index % num_batches_print == 0:
                print(f"                      minibatch#{batch_index:03d} = {loss_avg:.4f}")

    return model, losses


if __name__ == '__main__':
    # Load raw data into a HuggingFace Dataset 
    data_path = os.path.join(os.path.expanduser('~'), 'Downloads')
    text_field = torchtext.datasets.babi.BABI20Field(50)
    train_examples = torchtext.datasets.BABI20(data_path + '/tasks_1-20_v1-2/en-valid/qa4_train.txt', text_field) 
    train_dataset = utils.getDataset(train_examples)  #datasets.arrow_dataset.Dataset
    #Dataset({
    #    features: ['story', 'answer', 'query', 'input_ids', 'attention_mask', 'start_positions', 'end_positions'],
    #    num_rows: 900
    #})
    
    # list of lists
    #train_features = {key: train_ds[key] for key in ['input_ids', 'attention_mask']}
    #
    #train_labels = {"start_positions": tf.reshape(train_ds['start_positions'], shape=[-1,1]),
    #                'end_positions': tf.reshape(train_ds['end_positions'], shape=[-1,1])}
    
    selected_columns = ['input_ids','attention_mask', 'start_positions', 'end_positions']
    train_dataset.set_format(type='tf', columns=selected_columns)
    
    #>>> train_dataset[0]
    #{'input_ids': <tf.Tensor: shape=(26,), dtype=int64, numpy=
    #array([ 101, 1996, 2436, 2003, 2167, 1997, 1996, 3829, 1012, 1996, 3871,
    #       2003, 2148, 1997, 1996, 3829, 1012,  102, 2054, 2003, 2167, 1997,
    #       1996, 3829, 1029,  102])>, 
    #'attention_mask': <tf.Tensor: shape=(26,), dtype=int64, numpy=
    #array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #       1, 1, 1, 1])>, 
    #'start_positions': <tf.Tensor: shape=(), dtype=int64, numpy=2>, 
    #'end_positions': <tf.Tensor: shape=(), dtype=int64, numpy=2>}
    
    
    inputs = {'input_ids': train_dataset['input_ids'],
              'attention_mask': train_dataset['attention_mask']
              }
    #train_features = 'input_ids': 900 list of 30 lists
    
    labels = {'start_positions': tf.reshape(train_dataset['start_positions'], shape=[-1,1]),
              'end_positions': tf.reshape(train_dataset['end_positions'], shape=[-1,1])
              }

    #-------------------------------------------------- 
    random_suffix = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
    model_dirname = '01_27_24_{:s}'.format(random_suffix)
    hyperparams_filename = '01_27_24_{:s}.hparams'.format(random_suffix)
    hyperparams = {}
    hyperparams['batch_size'] = 16 
    hyperparams['num_epochs'] = 10 
    hyperparams['num_batches_print'] = 50
    hyperparams['learning_rate'] = 1e-5
    hyperparams['name'] = model_dirname

    inputs_and_labels = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(hyperparams['batch_size'])
    model, losses = buildModelAndTrain(hyperparams, inputs_and_labels)
    
    model.save_pretrained(os.path.join('../models/', model_dirname))
    hyperparams['history'] = losses 
    with open(os.path.join('../models/', hyperparams_filename), 'wb') as f:
        pickle.dump(hyperparams, f)

    os.system('mpg123 -q ~/Downloads/beep-05.mp3')
