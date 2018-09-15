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
parser.add_argument('content_image_path', metavar='base', type=str, help='Path to image to transform')
parser.add_argument('style_image_path', metavar='ref', type=str, help='Path to the style reference image')
parser.add_argument('result_prefix', metavar='res_prefix', type=str, help='Prefix for the saved results')
parser.add_argument('--iter', type=int, default=10, required=False, help='Number of iterations to run')
parser.add_argument('--content_weight', type=float, default=0.025, required=False, help='Content weight')
parser.add_argument('--style_weight', type=float, default=5.0, required=False, help='style weight')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False, help='total variation weight')

args = parser.parse_args()

#base, style and result image paths
content_image_path = args.content_image_path
style_image_path = args.style_image_path
result_prefix = args.result_prefix

#number of iterations, content and style weights
iterations = args.iter
content_weight = args.content_weight
style_weight = args.style_weight
tv_weight = args.tv_weight

# dimensions of the generated picture.
#width, height = load_img(base_image_path).size
img_nrows = 512
img_ncols = 512

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
content_image = K.variable(preprocess_image(content_image_path))
style_image = K.variable(preprocess_image(style_image_path))
combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

#creating tensor to be fed to the model
input_tensor = K.concatenate([content_image, style_image, combination_image], axis = 0)

#creating or defining the model and building it with pre-trained weights
model = vgg19.VGG19(input_tensor = input_tensor, weights = 'imagenet', include_top = False)
print('Loaded Model')

output_dict = dict([(layer.name, layer.output) for layer in model.layers])


#defining the gram matrix
def gram_matrix(x):
	features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))
	return gram


def style_cost(style, combination):
	S = gram_matrix(style)
	C = gram_matrix(combination)
	channels = 3
	size = img_nrows * img_ncols
	return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def content_loss(content, combination):
	return K.sum(K.square(combination - content))


def total_variation_loss(x):
	a = K.square(
		x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
	b = K.square(
		x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))

#initializing the loss value
loss = K.variable(0.)
#feature output of the layer block5_conv2
layer_features = output_dict['block2_conv2']
content_image_features = layer_features[0, :, :, :]
combination_image_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(content_image_features, combination_image_features)

feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']
for layer_name in feature_layers:
	layer_features = output_dict[layer_name]
	style_image_features = layer_features[1, :, :, :]
	combination_image_features = layer_features[2, :, :, :]
	loss += (style_weight / len(feature_layers)) * style_cost(style_image_features, combination_image_features)


grads = K.gradients(loss, combination_image)

outs = [loss]
outs += grads

f_outputs = K.function([combination_image], outs)


def eval_loss_and_grads(x):
	x = x.reshape((1, img_nrows, img_ncols, 3))
	out = f_outputs([x])
	loss_value = out[0]
	grad_values = out[1].flatten().astype('float64')
	return loss_value, grad_values


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()



x = np.random.uniform(0, 255, (1, img_nrows, img_ncols, 3)) - 128.

for i in range(iterations):
	print('Start of iteration', i)
	start_time = time.time()
	x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
	                                 fprime=evaluator.grads, maxfun=20)
	print('Current loss value:', min_val)
	# save current generated image
	img = deprocess_image(x.copy())
	fname = result_prefix + '_at_iteration_%d.png' % i
	save_img(fname, img)
	end_time = time.time()
	print('Image saved as', fname)
	print('Iteration %d completed in %ds' % (i, end_time - start_time))