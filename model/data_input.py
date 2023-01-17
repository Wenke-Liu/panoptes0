import tensorflow as tf
import numpy as np
import tensorflow.keras as keras


class DataSet:
    def __init__(self,
                 filenames,
                 labels,
                 tile_weights=None,
                 covariate=None,
                 img_size=299,
                 sym=True,
                 legacy=False
                 ):
        self.filenames = np.asarray(filenames)
        self.covariate = covariate 
        self.labels = labels
        self.tile_weights = tile_weights
        self.c_dim = None
        self.img_size = img_size
        self.sym = sym
        self.options = tf.data.Options()
        self.legacy = legacy

        self.options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        print('Number of samples: ' + str(len(self.labels)))
        if tile_weights is not None:
            print('Using tile weights.')
        if covariate is not None:
            self.covariate = np.asarray(self.covariate, dtype=np.float32)
            print('Including {} covariates.'.format(str(self.covariate.shape[1])))

    def create_dataset(self,
                       shuffle=True,
                       augmentation=False,
                       ds_epoch=100,
                       batch_size=8):

        dataset = tf.data.Dataset.from_tensor_slices((self.filenames[:, 0], self.filenames[:, 1], self.filenames[:, 2],
                                                      self.covariate, self.labels, self.tile_weights))
        dataset = dataset.with_options(self.options)
        if shuffle:
            dataset = dataset.shuffle(len(self.labels),
                                      reshuffle_each_iteration=True)  # perfect shuffle
        dataset = dataset.repeat(ds_epoch)
        dataset = dataset.map(lambda x1, x2, x3, z, y, w:
                              self.parse_function(filename1=x1, filename2=x2, filename3=x3,
                                                  cov=z, label=y, weight=w, augmentation=augmentation))   #num_parallel_calls=8
                                                  

        dataset = dataset.batch(batch_size)    #num_parallel_calls=8
        dataset = dataset.prefetch(2)
        print('Element spec: ' + str(dataset.element_spec))
        print('Augmentation: ' + str(augmentation))
        
        return dataset

    def parse_function(self,
                       filename1, filename2, filename3, cov, label, weight, augmentation=False):

        def read(fn):
            im_string = tf.io.read_file(fn)
            im = tf.image.decode_png(im_string, channels=3)

            if self.legacy:
                im = tf.reverse(im,axis=[-1])
                im = tf.cast(im,tf.float32) #maintains [0, 255)
            else:
                im = tf.image.convert_image_dtype(im, tf.float32) #will be changed to [0, 1)
            return im

        image1 = read(filename1)
        image2 = read(filename2)
        image3 = read(filename3)

        def augment(im):
            angles = tf.cast(tf.random.uniform([], 0, 4), tf.int32)
            im = tf.image.rot90(im, k=angles)
            im = tf.image.random_flip_left_right(im)
            im = tf.image.random_flip_up_down(im)
            #im = tf.image.random_jpeg_quality(im, 30, 100)
            im = tf.image.random_hue(im, max_delta=0.02)
            im = tf.image.random_brightness(im, max_delta=0.02)
            im = tf.image.random_contrast(im, lower=0.9, upper=1.1)
            im = tf.image.random_saturation(im, lower=0.9, upper=1.1)
            
            return im

        def clip_and_resize(im):
            im = tf.clip_by_value(im, 0.0, 1.0)
            im = tf.image.resize(im, [self.img_size, self.img_size])
            return im
        
        def symmetric(im):
            im = tf.subtract(im, 0.5)
            im = tf.multiply(im, 2.0)
            return im

        if augmentation:  # online training augmentation
            image1 = augment(image1)
            image2 = augment(image2)
            image3 = augment(image3)

        if not self.legacy:
            image1 = clip_and_resize(image1)
            image2 = clip_and_resize(image2)
            image3 = clip_and_resize(image3)

            if self.sym:
                image1 = symmetric(image1)
                image2 = symmetric(image2)
                image3 = symmetric(image3)
        
        if self.covariate is not None:
            xs = (image1, image2, image3, cov)
        else:
            xs = (image1, image2, image3)

        if self.tile_weights is not None:
            return xs, label, weight
        else:
            return xs, label