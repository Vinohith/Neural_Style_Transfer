from keras import backend as K


#defining the gram matrix
def gram_matrix(x):
	features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))
	return gram


#defining a function to calculate the style loss
#loss between the combination image and the style image
def style_loss(style, combination, img_nrows, img_ncols):
	S = gram_matrix(style)
	C = gram_matrix(combination)
	channels = 3
	size = img_nrows * img_ncols
	return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

#defining a function to calculate the style loss
#loss between the combination image and the actual content image
def content_loss(content, combination):
	return K.sum(K.square(combination - content))


#defining the total variation loss
#this is only to make the output combination image more smooth
def total_variation_loss(x, img_nrows, img_ncols):
	a = K.square(
		x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
	b = K.square(
		x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))