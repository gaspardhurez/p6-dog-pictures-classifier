from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def encode_dog_races(y, num_classes):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y = to_categorical(y_encoded, num_classes=num_classes)
    
    return y, label_encoder.classes_

