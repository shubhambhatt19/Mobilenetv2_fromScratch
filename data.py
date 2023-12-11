import tensorflow as tf
from IPython import embed


feature = {
    # 'image/height': tf.io.FixedLenFeature([], tf.int64),#dataset_util.int64_feature(height),
    # 'image/width': tf.io.FixedLenFeature([], tf.int64),#dataset_util.int64_feature(width),
    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),#dataset_util.bytes_feature(filename),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),#dataset_util.bytes_feature(encoded_jpg),
    # 'image/object/class/text': tf.io.FixedLenFeature([], tf.string),#dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),#dataset_util.int64_list_feature(classes)    
    }


def get_features(example_proto, feature = feature): 
    feature = tf.io.parse_single_example(example_proto, feature)
    image = tf.image.decode_image(feature["image/encoded"], channels=3)
    image.set_shape([224, 224, 3])
    image = tf.image.resize(image, (224, 224))
    label = feature["image/object/class/label"]
    # label = tf.cast(label, tf.int32)
    # label = tf.keras.utils.to_categorical(label)
    # label = tf.convert_to_tensor(label, dtype=tf.float32)
    
    
    return image, label
    

def parsed_data(train_record_path = None, test_record_path = None): 
    """input : tfrecord file name
       Output: parserd output of the tfrecord, i.e images and its labels"""
    train_dataset = tf.data.TFRecordDataset(train_record_path)
    parsed_train_dataset = train_dataset.map(get_features)

    val_dataset = tf.data.TFRecordDataset(test_record_path)
    parsed_val_dataset = val_dataset.map(get_features)
    return parsed_train_dataset, parsed_val_dataset