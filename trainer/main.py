from trainer.model import model_keras
from trainer.utils import data_manipulating, fix_flux, sigmoid_rld
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn import preprocessing
from tensorflow.python.lib.io import file_io
import argparse
import os
from keras.optimizers import SGD
import pickle


def main(job_dir, **args):
    # buraya data_dir koyucaz
    logs_path = 'gs://stone-door-214105.appspot.com/logs/'
    print(os.getcwd())

    model = model_keras(14)

    model.compile(optimizer=SGD(lr=0.1), loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])
    model.summary()
    with file_io.FileIO("gs://stone-door-214105.appspot.com/training_set_metadata.csv", mode='r') as f:
        meta = pd.read_csv(f)
    with file_io.FileIO("gs://stone-door-214105.appspot.com/training_set.csv", mode='r') as f:
        tr_set = pd.read_csv(f)
    tr_set, meta = data_manipulating(training_set=tr_set, metadata=meta)

    training = pd.merge(tr_set, meta, on="object_id")
    labels = training['target']
    training.drop(columns=['target', 'object_id'], inplace=True)

    training = training.values
    training = preprocessing.scale(training)
    labels = labels.values
    unq = np.unique(labels)
    labeler = {v: i for i, v in enumerate(unq)}
    labels = to_categorical([labeler[label] for label in labels])
    #    X_train, X_test, y_train, y_test = train_test_split(training, labels, test_size=0.2)

    del tr_set, meta

    model.fit(x=training, y=labels, epochs=60, batch_size=100)
    model.save("model.h5")
    with file_io.FileIO("model.h5", mode='rb')as input_f:
        with file_io.FileIO('gs://stone-door-214105.appspot.com/model.h5', mode="wb") as output_f:
            output_f.write(input_f.read())
    print("Training finished")
    test_set_files = ['test_set-000.csv', 'test_set-001.csv', 'test_set-002.csv', 'test_set-003.csv',
                      'test_set-004.csv', 'test_set-005.csv', 'test_set-006.csv', 'test_set-007.csv',
                      'test_set-008.csv', 'test_set-009.csv', 'test_set-010.csv', 'test_set-011.csv',
                      'test_set-012.csv', 'test_set-013.csv', 'test_set-014.csv', 'test_set-015.csv',
                      'test_set-016.csv', 'test_set-017.csv', 'test_set-018.csv', 'test_set-019.csv',
                      'test_set-020.csv', 'test_set-021.csv', 'test_set-022.csv', 'test_set-023.csv',
                      'test_set-024.csv', 'test_set-025.csv', 'test_set-026.csv', 'test_set-027.csv',
                      'test_set-028.csv', 'test_set-029.csv', 'test_set-030.csv', 'test_set-031.csv',
                      'test_set-032.csv', 'test_set-033.csv', 'test_set-034.csv', 'test_set-035.csv',
                      'test_set-036.csv', 'test_set-037.csv', 'test_set-038.csv', 'test_set-039.csv',
                      'test_set-040.csv', 'test_set-041.csv', 'test_set-042.csv', 'test_set-043.csv',
                      'test_set-044.csv', 'test_set-045.csv']
    with file_io.FileIO('gs://stone-door-214105.appspot.com/test_set_metadata.csv', mode='r') as f:
        meta_test = pd.read_csv(f)

    for test_set_filepath in test_set_files:
        with file_io.FileIO(os.path.join('gs://stone-door-214105.appspot.com/', test_set_filepath), mode='r') as f:
            test_set = pd.read_csv(f)
        unqs_ids = np.unique(test_set['object_id'])
        new_meta = meta_test[meta_test['object_id'].isin(unqs_ids)]
        test_set, new_meta = data_manipulating(test_set, new_meta)
        testing = pd.merge(test_set, new_meta, on="object_id")
        del test_set, new_meta
        o_id = testing['object_id']
        o_id = np.array(o_id)
        testing.drop(columns=['object_id'], inplace=True)
        testing = preprocessing.scale(testing)
        prediction = model.predict(testing)
        o_id = o_id.reshape(len(o_id), 1)
        prediction = np.append(prediction, o_id, axis=1)
        if test_set_filepath == test_set_files[0]:
            a = pd.DataFrame(prediction)
            a = a.transpose()
        else:
            m = pd.DataFrame(prediction)
            m = m.transpose()
            a = a.append(m, ignore_index=True)
        print("{}. set of preds complete".format(test_set_files.index(test_set_filepath)))

    a.columns = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52',
                 'class_53', 'class_62', 'class_64', 'class_65', 'class_67',
                 'class_88', 'class_90', 'class_92', 'class_95', 'object_id']
    a = a['object_id', 'class_6', 'class_15', 'class_16', 'class_42', 'class_52',
          'class_53', 'class_62', 'class_64', 'class_65', 'class_67',
          'class_88', 'class_90', 'class_92', 'class_95']

    a.to_pickle("submission.pkl")
    with file_io.FileIO('submission.pkl', mode="rb") as inp_fi:
        with file_io.FileIO('gs://stone-door-214105.appspot.com/' + 'submission.pkl', mode="wb") as out_fi:
            pickle.dump(inp_fi, out_fi, protocol=2)
    a['class_99'] = np.zeros(a['object_id'].values.shape)
    a.to_pickle("submission_99.pkl")
    with file_io.FileIO('submission_99.pkl', mode="rb") as inp_fi:
        with file_io.FileIO('gs://stone-door-214105.appspot.com/' + 'submission_99.pkl', mode="wb") as out_fi:
            pickle.dump(inp_fi, out_fi, protocol=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    main(arguments)
