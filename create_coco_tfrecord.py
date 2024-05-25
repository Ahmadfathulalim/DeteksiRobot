import tensorflow as tf
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.dataset_tools import create_coco_tf_record
import os

flags = tf.compat.v1.app.flags
flags.DEFINE_string('train_image_dir', 'coco_dataset/train2017', 'Training image directory.')
flags.DEFINE_string('val_image_dir', 'coco_dataset/val2017', 'Validation image directory.')
flags.DEFINE_string('train_annotations_file', 'coco_dataset/annotations/instances_train2017.json', 'Training annotations JSON file.')
flags.DEFINE_string('val_annotations_file', 'coco_dataset/annotations/instances_val2017.json', 'Validation annotations JSON file.')
flags.DEFINE_string('output_dir', 'coco_tfrecord', 'Output data directory.')

FLAGS = flags.FLAGS

def main(_):
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    create_coco_tf_record._create_tf_record_from_coco_annotations(
        FLAGS.train_annotations_file,
        FLAGS.train_image_dir,
        os.path.join(FLAGS.output_dir, 'coco_train.record'))

    create_coco_tf_record._create_tf_record_from_coco_annotations(
        FLAGS.val_annotations_file,
        FLAGS.val_image_dir,
        os.path.join(FLAGS.output_dir, 'coco_val.record'))

if __name__ == '__main__':
    tf.compat.v1.app.run()
