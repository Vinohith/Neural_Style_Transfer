import imageio
import sys


def create_gif(filenames, duration):
	images = []
	for filename in filenames:
		images.append(imageio.imread(filename))
	imageio.mimsave('movie.gif', images, duration=duration)


script = sys.argv.pop(0)

duration = float(sys.argv.pop(0))

filenames = sys.argv

create_gif(filenames, duration)