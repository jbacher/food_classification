import json
import numpy as np
import sys
import os
from flask import Flask
from firebase import firebase

caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')

import caffe

app = Flask(__name__)

#firebase connection
firebase = firebase.FirebaseApplication('https://project-7670672465659046853.firebaseio.com')
result = firebase.get('/data',None)

@app.route("/api/classify")

def hello():
    img_name = result['food_post']
    caffe.set_mode_cpu()
    model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
    model_weights = caffe_root+ 'models/bvlc_reference_caffenet/caffenet_train_iter_4078.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)
    #transform the image

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    net.blobs['data'].reshape(50,3,227,227)

    image = caffe.io.load_image(img_name) #load image
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward() #classifying the image
    output_prob = output['prob'][0]
    top_inds = output_prob.argsort()[::-1][:5]
    json_file = open('/media/bacherj/3d29b762-2e94-4cf0-bdd5-d13d4de1ae7b/cat_words.json', 'r');
    data = json.load(json_file)

    for i in top_inds:
        print(str(output_prob[i]) + ' ' + str(data[i].values()[0]))
    return "success"

if __name__ == "__main__":
    app.run()
