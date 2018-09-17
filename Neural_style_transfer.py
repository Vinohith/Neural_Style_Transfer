#importing necessary libraries
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications import vgg19
from keras import backend as K
import numpy as np
from losses import style_loss, content_loss, total_variation_loss
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
parser.add_argument('--total_variation_weight', type=float, default=1.0, required=False, help='total variation weight')

args = parser.parse_args()


#base, style and result image paths
content_image_path = args.content_image_path
style_image_path = args.style_image_path
result_prefix = args.result_prefix


#number of iterations, content, style and total variation weights
iterations = args.iter
content_weight = args.content_weight
style_weight = args.style_weight
total_variation_weight = args.total_variation_weight


#initializing dimensions of the generated picture.
img_nrows = 512
img_ncols = 512


#loadinng and preprocessing the image
def preprocess_image(image_path):
	img = load_img(image_path, target_size=(img_nrows, img_ncols))
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = vgg19.preprocess_input(img)
	return img


#deprocessing the output image
def deprocess_image(x):
	x = x.reshape((img_nrows, img_ncols, 3))
	x = x[:, :, ::-1]
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68	
	x = np.clip(x, 0, 255).astype('uint8')
	return x


#initializing the content and style images as a tensorflow variable with 
#the perprocessed content and style image
content_image = K.variable(preprocess_image(content_image_path))
style_image = K.variable(preprocess_image(style_image_path))

#creating a tensorflow placeholder for the combination image
combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

#creating tensor to be fed to the model(0=>content_image, 1=>style_image, 2=>combination_image along 0 axis)
input_tensor = K.concatenate([content_image, style_image, combination_image], axis = 0)

#creating or defining the model and building it with pre-trained weights
model = vgg19.VGG19(input_tensor = input_tensor, weights = 'imagenet', include_top = False)
print('Loaded Model')

#creating a dictionary with the layer name as key and the layer output as value
output_dict = dict([(layer.name, layer.output) for layer in model.layers])


#initializing the loss as a tensorflow variable
loss = K.variable(0.)

layer_features = output_dict['block2_conv2']
#content_image_features output at block2_conv2
content_image_features = layer_features[0, :, :, :]
#combination_image_features output at block2_conv2
combination_image_features = layer_features[2, :, :, :]
#calculating the content loss
loss += content_weight * content_loss(content_image_features, combination_image_features)


#layers at which the style loss is to be calculated
feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']


#iterating over the feature layers
for layer_name in feature_layers:
	layer_features = output_dict[layer_name]
	#style_image_features output at the particular feature layer
	style_image_features = layer_features[1, :, :, :]
	#combination_image_features output at the particular feature layer
	combination_image_features = layer_features[2, :, :, :]
	#calculating the style loss
	sl = style_loss(style_image_features, combination_image_features, img_nrows, img_ncols)
	loss += (style_weight / len(feature_layers)) * sl

#calculating the total variation loss in the combination image
loss += total_variation_weight * total_variation_loss(combination_image, img_nrows, img_ncols)


#getting the gradients of the loss w.r.t the pixels of the combination image
grads = K.gradients(loss, combination_image)

outs = [loss]
outs += grads

#instantiating a Keras function
f_outputs = K.function([combination_image], outs)


#defining the function to evaluate loss and gradients
def eval_loss_and_grads(x):
	x = x.reshape((1, img_nrows, img_ncols, 3))
	out = f_outputs([x])
	loss_value = out[0]
	grad_values = out[1].flatten().astype('float64')
	return loss_value, grad_values


#the Evaluator makes it possible to compute 
#loss and gradients in one pass. This because
#the used optimizer function reuires both the loss 
#and gradients
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


#random combination to start of with
x = np.random.uniform(0, 255, (1, img_nrows, img_ncols, 3)) - 128.


#running for the specified number of iterations
for i in range(iterations):
	print('Start of iteration', i)
	start_time = time.time()
	#minimizing the loss using fmin_l_bfgs_b
	x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
	                                 fprime=evaluator.grads, maxfun=20)
	print('Current loss value:', min_val)
	img = deprocess_image(x.copy())
	fname = result_prefix + '_at_iteration_%d.png' % i
	#saving the output combination image
	save_img(fname, img)
	end_time = time.time()
	print('Image saved as', fname)
	print('Iteration %d completed in %ds' % (i, end_time - start_time))