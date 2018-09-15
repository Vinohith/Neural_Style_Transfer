#importing libraries
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications import vgg19
from keras import backend as K
import numpy as np
import time
import argparse
from scipy.optimize import fmin_l_bfgs_b

#various arguments
parser = argparse.ArgumentParser(description = 'Neural Style Transfer')
parser.add_argument('base_image_path', metavar='base', type=str, help='Path to image to transform')
parser.add_argument('style_reference_image_path', metavar='ref', type=str, help='Path to the style reference image')
parser.add_argument('result_prefix', metavar='res_prefix', type=str, help='Prefix for the saved results')
parser.add_argument('--iter', type=int, default=10, required=False, help='Number of iterations to run')
parser.add_argument('--content_weight', type=float, default=0.025, required=False, help='Content weight')
parser.add_argument('--style_weight', type=float, default=1.0, required=False, help='style_weight')

args = parser.parse_args()

#base, style and result image paths
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix

#number of iterations, content and style weights
iterations = args.iter
content_weight = args.content_weight
style_weight = args.style_weight

# dimensions of the generated picture.
width, height = load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

def preprocess_image(image_path):
	img = load_img(image_path, target_size=(img_nrows, img_ncols))
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = vgg19.preprocess_input(img)
	return img


def deprocess_image(x):
	x = x.reshape((3, img_nrows, img_ncols))
	x = x.transpose((1,2,0))
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 0] += 123.68
	#BGR to RGB
	x = x[:, :, ::-1]
	x = np.clip(x, 0, 255).astype('uint8')
	return x


#tensor representation of images
base_image = K.variable(preprocess_image(base_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))
combination_image = K.placeholder((1, 3, img_nrows, img_ncols))

#creating tensor to be fed to the model
input_tensor = K.concatenate([base_image, style_reference_image, combination_image], axis = 0)

#creating or defining the model and building it with pre-trained weights
model = vgg19.vgg19(input_tensor = input_tensor, weights = 'imagenet', include_top = False)
print('Loaded Model')

output_dict = dict([(layer.name, layer.output) for layer in model.layers])


#defining the gram matrix
def gram_matrix(x):
	features = K.batch_flatten(x)
	gram = K.dot(features, K.transpose(features))
	return gram


def style_cost(style, combination):
	S = gram_matrix(style)
	C = gram_matrix(combination)
	channels = 3
	size = img_nrows * img_ncols
	return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def content_loss(base, combination):
	return K.sum(K.square(combination - base))


