
###############


import tensorflow as tf
import math


##############


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)



def leaky_relu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * tf.abs(x)
     


def Ramp(x):
    return tf.minimum(tf.maximum(0., x),1.)


##############


#generate G and D models
class GenDisc(object):
    
    def __init__(self,disc,dims, out_dims,dims_nCross=0, soh=False, ohFeatureIndices=None, catEmbeddingDims=None, numColsLen=None, reg_scale=None,
                 use_dropout_everywhere=False,keep_prob=False,useResNet=False,
                 nSchortcut=None):
        
        #generate G or D, depending on value or disc variable
        self.disc = disc
        if disc:
            self.name = 'discriminator'
        else:
            self.name = 'generator'


        self.dims = dims        
        self.out_dims = out_dims#1 for wgan, bigger for cramer
        self.reg_scale = reg_scale
        self.dims_nCross = dims_nCross
        self.use_dropout_everywhere = use_dropout_everywhere
        self.keep_prob = keep_prob
        self.useResNet = useResNet
        self.nSchortcut = nSchortcut
        self.soh = soh
        self.ohFeatureIndices = ohFeatureIndices
        self.catEmbeddingDims = catEmbeddingDims
        self.numColsLen = numColsLen


    def __call__(self, inputs, reuse=True):
             
        with tf.variable_scope(self.name, reuse=reuse) as scope:
            if self.reg_scale is not None:
                scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=self.reg_scale))
                
            
            ndim = inputs.get_shape().ndims
            input_width = inputs.get_shape()[ndim-1].value

               
            stddev = 1.0 / math.sqrt(input_width)
            norm = tf.truncated_normal_initializer(stddev=stddev)
            const = tf.constant_initializer(0.0)

            
            if self.disc and self.catEmbeddingDims is not None:
                #assuming numeric variables are first
                num_inputs = inputs[:,:self.numColsLen]
                cat_inputs =  inputs[:,self.numColsLen:]                
                embedded_cat_inputs=[]

                start = 0
                for i,end in enumerate(self.ohFeatureIndices):         
                    cat_i=cat_inputs[:,start:end]
                    shape = [cat_i.get_shape()[ndim-1].value, self.catEmbeddingDims[i]]
                    W = tf.get_variable("W"+str(i), shape=shape, 
                                        initializer=tf.random_normal_initializer(stddev=xavier_init(shape)))                                        
                    embedded_cat_inputs.append(tf.matmul(cat_i,W))
                
                                                                    
                inputs = tf.concat(axis=1, values=[num_inputs] + embedded_cat_inputs)


            #cross-net layers
            tempVec = inputs
            input_widthCross=inputs.get_shape()[ndim-1].value
            for i in range(self.dims_nCross):
                with tf.variable_scope('crossLayer'+str(i)):
                    w = tf.get_variable(name='weights', shape=[input_widthCross], dtype=tf.float32, initializer=norm)
                    b = tf.get_variable(name='biases', shape=[input_widthCross], dtype=tf.float32, initializer=const)
                    x0xt = tf.expand_dims(inputs, -1) * tf.expand_dims(tempVec, -2)
                    out= tf.tensordot(x0xt, w, [[ndim-1], [0]]) + tempVec + b
               
                tempVec = out
                if self.use_dropout_everywhere:    
                    with tf.name_scope("dropout"+str(i)):
                        tempVec = tf.nn.dropout(tempVec, self.keep_prob)
                
                
                
            ##fc layers
            tempVec_fc = inputs
            depth=0
            shortcutInput = inputs
            shortcutDim = input_width

            for i, dim in enumerate(self.dims):
                out_hidden =tf.layers.dense(tempVec_fc, dim,
                                            kernel_initializer=tf.random_normal_initializer(stddev=xavier_init([tempVec_fc.get_shape()[ndim-1].value, dim])),
                                            bias_initializer=tf.constant_initializer())#no activation
                depth=depth+1
                
                if self.useResNet and depth==self.nSchortcut:
                    depth=0
                    if dim==shortcutDim:
                        tempVecResize=shortcutInput
                    else:
                        newDim=dim
                        tempVecResize=self.resNet_resizeInput(shortcutInput,newDim,rezType="project",scope="ProjectD"+str(i),reuse_scope=reuse)
                    
                    tempVec_fc = leaky_relu(out_hidden + tempVecResize)
                    if self.use_dropout_everywhere:    
                        with tf.name_scope("dropoutD"+str(i)):
                            tempVec_fc = tf.nn.dropout(tempVec_fc, self.keep_prob)
                    
                    shortcutInput=tempVec_fc
                    shortcutDim=dim
                    
                else:
                    tempVec_fc = leaky_relu(out_hidden)
                    if self.use_dropout_everywhere:    
                        with tf.name_scope("dropoutD"+str(i)):
                            tempVec_fc = tf.nn.dropout(tempVec_fc, self.keep_prob)
               
                
                
            if self.dims_nCross>0:        
                cat_res=tf.concat([tempVec,tempVec_fc],axis=1)#concat by column
            else:
                cat_res=tempVec_fc
            
            output = tf.layers.dense(cat_res, self.out_dims)
            

          
            if self.disc or self.ohFeatureIndices is None:
                prob = tf.nn.sigmoid(output)
                
            else: #softmax on each categorical feature
                num_outputs = output[:,:self.numColsLen]
                cat_outputs =  output[:,self.numColsLen:]
                #take softmax on groups of cols representing categoricals
                out_cat_list=[]
                start = 0
                for i,end in enumerate(self.ohFeatureIndices):
                    out_cat_list.append(tf.nn.softmax(cat_outputs[:,start:end]))
                    start = end

                prob=tf.concat(axis=1, values=[tf.nn.sigmoid(num_outputs)] + out_cat_list)

            return prob,output
        


    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

