import numpy as np 
from PIL import Image 
import skimage
import json
import base64
from cStringIO import StringIO
import sys
caffe_root = '/home/oleksandr/Caffe_GPU/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

class CaffeGoogleNetPlaces:

	def __init__(self):

		model_def = '/home/oleksandr/Documents/Python_Scripts/KUB/googlenet_places205/deploy_places205.protxt'
		model_weights = '/home/oleksandr/Documents/Python_Scripts/KUB/googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel'

		self.net = caffe.Net(model_def,      # defines the structure of the model
		                model_weights,  # contains the trained weights
		                caffe.TEST)     # use test mode (e.g., don't perform dropout)


		from caffe.proto import caffe_pb2
		# load the mean for subtraction
		mean_blob = caffe_pb2.BlobProto()
		with open('/home/oleksandr/Documents/Python_Scripts/KUB/hybridCNN_mean.binaryproto') as f:
		    mean_blob.ParseFromString(f.read())
		mu = np.asarray(mean_blob.data, dtype=np.float32).reshape(
		    (mean_blob.channels, mean_blob.height, mean_blob.width))

		# mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
		mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
		#print 'mean-subtracted values:', zip('BGR', mu)

		# create transformer for the input called 'data'
		self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

		self.transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
		self.transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
		self.transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
		self.transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


		# set the size of the input (we can skip this if we're happy
		#  with the default; we can also change it later, e.g., for different batch sizes)
		self.net.blobs['data'].reshape(1,        # batch size
		                          3,         # 3-channel (BGR) images
		                          224, 224)  # image size is 224x224


	def get_features(self, img):

		caffe.set_mode_gpu()
        
		#image = caffe.io.load_image(image_path)
		image64 = img.get('photo').decode('base64')
		image_str = StringIO(image64)
		im_arr = np.asarray(Image.open(image_str))

		img_float = skimage.img_as_float(im_arr).astype(np.float32)

		self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img_float)

		output = self.net.forward()

		return [(output['prob'][0]).argmax(), self.net.blobs['pool5/7x7_s1'].data.squeeze()]



	def get_features_path(self, img):

		caffe.set_mode_gpu()
        
		image = caffe.io.load_image(img)

		self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)

		output = self.net.forward()

		return [(output['prob'][0]).argmax(), self.net.blobs['pool5/7x7_s1'].data.squeeze()]