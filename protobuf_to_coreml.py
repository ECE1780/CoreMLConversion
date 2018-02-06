import tfcoreml

# To go from protobuf -> CoreML model, just call
# a function and specify the inputs/outputs

coreml_model = tfcoreml.convert(
    tf_model_path='model.pb',
    mlmodel_path='model.mlmodel',
    input_name_shape_dict={'image:0': [1, 28, 28, 1]},
    output_feature_names=['prediction:0'])
