import tensorflow as tf
import os

# TensorFlow checkpoint saves (.meta, .data) keep separate information
# about the graph definition and the variables, respectively
#
# Much of that information is not necessary when you want to
# use your model in production (e.g. space for the gradients of variables
# during gradient descent, the name of some intermediate node, etc.)
#
# Convert to a protobuf (.pb), which will contain the graph definition
# and constant values for the weights/biases in a single file


def freeze_graph(model_dir, model_name, output_node):
    # Restore the computation graph of our model
    saver = tf.train.import_meta_graph(os.path.join(model_dir, model_name+'.ckpt.meta'))
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        # Restore all weights/biases
        saver.restore(sess, os.path.join(model_dir, model_name+'.ckpt'))

        # Store the variables as constants in the graph, and save to file
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, [output_node])
        with tf.gfile.GFile(model_name+'.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())


freeze_graph('saved_models', 'model', 'prediction')
