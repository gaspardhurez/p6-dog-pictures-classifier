import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt

# Définir l'architecture du modèle CNN
def create_cnn_model(input_shape, num_classes, hp=None):
    model = models.Sequential()
    
    # Nombre de filtres pour chaque couche de convolution
    conv_1_filters = hp.Int('conv_1_filters', 32, 128, 32) if hp else 32
    conv_2_filters = hp.Int('conv_2_filters', 64, 256, 64) if hp else 64
    conv_3_filters = hp.Int('conv_3_filters', 128, 512, 128) if hp else 128
    dense_units = hp.Int('dense_units', 64, 256, 64) if hp else 128
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log') if hp else 1e-3

    # Construction du modèle avec les valeurs définies
    model.add(layers.Conv2D(conv_1_filters, (3, 3), activation='relu', input_shape=(input_shape)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(conv_2_filters, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(conv_3_filters, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def optimize_cnn_hyperparams(X_train, y_train, X_test, y_test, input_shape, num_classes, hp=None):

    def model_builder(hp):
        return create_cnn_model(input_shape, num_classes, hp)

    tuner = kt.RandomSearch(
        model_builder,  # Passez une fonction qui encapsule create_cnn_model
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='../logs',
        project_name='cnn_hyperparam_optimization'
    )

    tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    best_model = tuner.get_best_models(num_models=1)[0]

    return best_model



