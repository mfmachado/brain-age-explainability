import argparse
import os.path
from pathlib import Path

import tensorflow as tf
import numpy as np
import nibabel as nib

import saliency.core as saliency

class_idx_str = 'class_idx_str'


def call_model_function(images, call_model_args=None, expected_keys=None):
    model = call_model_args["model"]
    images = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
        if expected_keys == [saliency.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            _, output_layer = model(images)
            gradients = np.array(tape.gradient(output_layer, images))
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            conv_layer, output_layer = model(images)
            gradients = np.array(tape.gradient(output_layer, conv_layer))
            return {saliency.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                    saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Model path", type=str)
    parser.add_argument("image_path", type=str, help="Path of the image")
    parser.add_argument("output_path", type=str, help="Output path of the image")
    parser.add_argument("noise", help="SmoothGrad noise percentage", type=int)
    parser.add_argument("prefix_folder_save_exp_map", help="Prefix folder save explainability map", type=str)
    return parser.parse_args()


if __name__ == '__main__':

    logger = Logger().get(Logger.NAME)

    args = parse_args()
    path_image = args.image_path
    noise = args.noise
    model_path = args.model_path
    path_save_exp_map = args.output_path

    parameters_exp_alg = {"nsamples": 25, "stdev_spread": noise/100}

    column_path = "path_abs"


    img_info = nib.load(path_image)
    im = img_info.get_fdata()
    logger.info("Generating explainability map for the image: " + row_file[column_path])

    all_tissue_predictions = []

    model = keras.models.load_model(model_path)

    conv_layer = model.get_layer('regression-block2_conv')
    model_original = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

    _, predictions = model_original(np.array([im]))
    prediction_class = predictions[0][0]
    call_model_args = {"model": model_original}


    # Construct the saliency object. This alone doesn't do anything.
    gradient_saliency = saliency.GradientSaliency()
    explainability_maps_img = gradient_saliency.GetSmoothedMask(im, call_model_function, call_model_args,
                                                                **parameters_exp_alg)

    nib.save(nib.Nifti1Image(explainability_maps_img, affine=img_info.affine), path_save_exp_map)
