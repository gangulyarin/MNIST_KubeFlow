# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Data exploration libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import kfp
import kfp.dsl as dsl
import kfp.components as comp

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

unique_train_labels = np.unique(train_labels)

train_images = train_images / 255.0

test_images = test_images / 255.0


def train(data_path, model_file):
    
    # func_to_container_op requires packages to be imported inside of the function.
    import pickle
    
    import tensorflow as tf
    from tensorflow.python import keras
    
    # Download the dataset and split into training and test data. 
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalize the data so that the values all fall between 0 and 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Define the model using Keras.
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Run a training job with specified number of epochs
    model.fit(train_images, train_labels, epochs=10)

    # Evaluate the model and print the results
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('Test accuracy:', test_acc)

    # Save the model to the designated 
    model.save(f'{data_path}/{model_file}')

    # Save the test_data as a pickle file to be used by the predict component.
    with open(f'{data_path}/test_data', 'wb') as f:
        pickle.dump((test_images,test_labels), f)


def predict(data_path, model_file, image_number):
    
    # func_to_container_op requires packages to be imported inside of the function.
    import pickle

    import tensorflow as tf
    from tensorflow import keras

    import numpy as np
    
    # Load the saved Keras model
    model = keras.models.load_model(f'{data_path}/{model_file}')

    # Load and unpack the test_data
    with open(f'{data_path}/test_data','rb') as f:
        test_data = pickle.load(f)
    # Separate the test_images from the test_labels.
    test_images, test_labels = test_data
    # Define the class names.
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Define a Softmax layer to define outputs as probabilities
    probability_model = tf.keras.Sequential([model, 
                                            tf.keras.layers.Softmax()])

    # See https://github.com/kubeflow/pipelines/issues/2320 for explanation on this line.
    image_number = int(image_number)

    # Grab an image from the test dataset.
    img = test_images[image_number]

    # Add the image to a batch where it is the only member.
    img = (np.expand_dims(img,0))

    # Predict the label of the image.
    predictions = probability_model.predict(img)

    # Take the prediction with the highest probability
    prediction = np.argmax(predictions[0])

    # Retrieve the true label of the image from the test labels.
    true_label = test_labels[image_number]
    
    class_prediction = class_names[prediction]
    confidence = 100*np.max(predictions)
    actual = class_names[true_label]
    
    
    with open(f'{data_path}/result.txt', 'w') as result:
        result.write(" Prediction: {} | Confidence: {:2.0f}% | Actual: {}".format(class_prediction,
                                                                        confidence,
                                                                        actual))
    
    print('Prediction has be saved successfully!')

# Create train and predict lightweight components.
train_op = comp.func_to_container_op(train, base_image='tensorflow/tensorflow:latest-gpu-py3')
predict_op = comp.func_to_container_op(predict, base_image='tensorflow/tensorflow:latest-gpu-py3')

# Define the pipeline
@dsl.pipeline(
   name='MNIST Pipeline',
   description='A toy pipeline that performs mnist model training and prediction.'
)

# Define parameters to be fed into pipeline
def mnist_container_pipeline(
    data_path: str,
    model_file: str, 
    image_number: int
):
    
    # Define volume to share data between components.
    vop = dsl.VolumeOp(
    name="create_volume",
    resource_name="data-volume", 
    size="1Gi", 
    modes=dsl.VOLUME_MODE_RWM)
    
    # Create MNIST training component.
    mnist_training_container = train_op(data_path, model_file) \
                                    .add_pvolumes({data_path: vop.volume})

    # Create MNIST prediction component.
    mnist_predict_container = predict_op(data_path, model_file, image_number) \
                                    .add_pvolumes({data_path: mnist_training_container.pvolume})
    
    # Print the result of the prediction
    mnist_result_container = dsl.ContainerOp(
        name="print_prediction",
        image='library/bash:4.4.23',
        pvolumes={data_path: mnist_predict_container.pvolume},
        arguments=['cat', f'{data_path}/result.txt']
    )

if __name__=="__main__":
    kfp_endpoint=None
    client = kfp.Client(host=kfp_endpoint)
    DATA_PATH = '/mnt'
    MODEL_PATH='mnist_model.h5'
    # An integer representing an image from the test set that the model will attempt to predict the label for.
    IMAGE_NUMBER = 0
    pipeline_func = mnist_container_pipeline
    experiment_name = 'fashion_mnist_kubeflow'
    run_name = pipeline_func.__name__ + ' run'

    arguments = {"data_path":DATA_PATH,
                "model_file":MODEL_PATH,
                "image_number": IMAGE_NUMBER}

    # Compile pipeline to generate compressed YAML definition of the pipeline.
    kfp.compiler.Compiler().compile(pipeline_func,  
    '{}.zip'.format(experiment_name))

    # Submit pipeline directly from pipeline function
    run_result = client.create_run_from_pipeline_func(pipeline_func, 
                                                    experiment_name=experiment_name, 
                                                    run_name=run_name, 
                                                    arguments=arguments)