
###############


import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
    
import model
from encode_decode import Data,classifier_filter, crossEvaluation


###############


os.environ['CUDA_VISIBLE_DEVICES'] = "1"
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


flags = tf.flags
flags.DEFINE_string("outFile", 'out-{}.pkl'.format(datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')),                # .1, .04
                   "outputFileName")
flags.DEFINE_string("traindir", './', "Directory to save files.")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("evaluate_every", 500, "evaluate every x batches")
tf.flags.DEFINE_integer("interv", 15000, "data generation frequency in iterations")



flags.DEFINE_integer("max_epoch", 1,                
                   "Number of epochs before stopping training.")
flags.DEFINE_integer("batch_size",256,               
                   "Batch size.")


flags.DEFINE_float("learning_rate", 0.0001,             
                   "Learning rate")
flags.DEFINE_float("LAMBDA", 10.0 ,
                   "weight gradient penalty term in D loss")


flags.DEFINE_float("init_scale", 0.05,                
                   "initial scale of the weights")
flags.DEFINE_float("reg_scale", 0.00001,      
                   "L2 regularization scale")               
flags.DEFINE_boolean("disable_l2_regularizer", True,
                   "disable L2 regularization on weights")



flags.DEFINE_boolean("z_normal", False,               
                   "use z from normal or uniform distribution")
flags.DEFINE_integer("zDim", 10,             
                   "dimension of z")


flags.DEFINE_boolean("use_dropout_everywhere", False,
                   "use droput between all layers")
flags.DEFINE_float("keep_prob", 0.5,                 
                   "Keep probability. 1.0 disables dropout")


flags.DEFINE_string("genDims_str", '(128,128)',                
                   "size FC layers used in G")
flags.DEFINE_string("discDims_str", '(128,128)',                
                   "size FC layers used in D")
flags.DEFINE_integer("discDims_nCross", 0,
                   "number cross-net layers")
flags.DEFINE_integer("discriminatorOut_dims", 64,               
                   "discriminator output dims, the biger the better for Cramer, 1 for WGAN")
flags.DEFINE_integer("n_critics", 5,                
                   "number of iterations of D")
       

flags.DEFINE_boolean("useResNet", False, 
                   "resnet conections betwween FC layers")
flags.DEFINE_integer("nSchortcut", 2,
                   "resnet conections betwween FC layers, every n layers")



flags.DEFINE_integer("emb_vocab_min", 5,
                   "minimum token count to have its own embeding")
flags.DEFINE_integer("k", 5,
                   "embedings dimension")


FLAGS = flags.FLAGS


############

def evaluateResults():

    fIn_real='./data/test.csv'    
    fIn_fake="./generated_data/result1/result.csv"


    df_real=pd.read_csv(fIn_real)
    df_final=pd.read_csv(fIn_fake) 
    n_subS=min(df_real.shape[0],df_final.shape[0])
    df_final=df_final.sample(n=n_subS,random_state =0)  
    
    
    cols_dict={'Origin_Country':'cat',
                   'Destination_Country':'cat',
                   'OfficeIdCountry':'cat',
                   'SaturdayStay':'cat',
                   'PurchaseAnticipation':'num',
                   'NumberPassenger':'cat',
                   'StayDuration':'num',
                   'Title':'cat',
                   'TravelingWithChild':'cat',
                   'ageAtPnrCreation':'num',
                   'Nationality2':'cat',
                   'BL':'cat'}
            


    #univariate evaluation: country origin
    c="Origin_Country"
    if cols_dict[c]=='cat':
        print(df_real[c].astype(str).value_counts(normalize=True).head(15))
        print(df_final[c].astype(str).value_counts(normalize=True).head(15))
    else:
        print(df_real[c].astype(np.float).describe())
        print(df_final[c].astype(np.float).describe())
        


    
    
    #univariate evaluation: age histogram
    x = df_final.ageAtPnrCreation.values
    y = df_real.ageAtPnrCreation.values
    xweights = 100 * np.ones_like(x) / x.size
    yweights = 100 * np.ones_like(y) / y.size    
    fig, ax = plt.subplots()
    ax.hist(x, weights=xweights, color='blue', alpha=0.5)
    ax.hist(y, weights=yweights, color='red', alpha=0.5)    
    ax.set(title='Histogram Comparison', ylabel='% of Dataset in Bin')
    ax.margins(0.05)
    ax.set_ylim(bottom=0)
    plt.show()     
    
    
    
    #univariate evaluation: stay duration for buisnes/leasure travellers
    #buisines
    x_b=df_real.StayDuration[df_real.BL=='B']
    y_b=df_final.StayDuration[df_final.BL=='B']
    #leasure
    x_l=df_real.StayDuration[df_real.BL=='L']
    y_l=df_final.StayDuration[df_final.BL=='L']
    
    
    xweights = 100 * np.ones_like(x_b) / x_b.size
    yweights = 100 * np.ones_like(y_b) / y_b.size
    fig, ax = plt.subplots()
    ax.hist(x_b, weights=xweights, color='blue', alpha=0.5)
    ax.hist(y_b, weights=yweights, color='red', alpha=0.5)
    ax.set(title='Histogram Comparison', ylabel='% of Dataset in Bin')
    ax.margins(0.05)
    ax.set_ylim(bottom=0)
    plt.show() 
     
    
    xweights = 100 * np.ones_like(x_l) / x_l.size
    yweights = 100 * np.ones_like(y_l) / y_l.size
    fig, ax = plt.subplots()
    ax.hist(x_l, weights=xweights, color='blue', alpha=0.5)
    ax.hist(y_l, weights=yweights, color='red', alpha=0.5)
    ax.set(title='Histogram Comparison', ylabel='% of Dataset in Bin')
    ax.margins(0.05)
    ax.set_ylim(bottom=0)
    plt.show() 
    
    
    
    
    #Multivariate evaluation: RF calssifier to separate real/synthetic data    
    df_real_aux=df_real.head(df_final.shape[0]).copy(deep=True)
    df_fake=df_final.copy(deep=True)

    col_target='label'
    cols_toOneHot=[c for c in cols_dict if cols_dict[c]=='cat']
    cols_featuresNum=[c for c in cols_dict if cols_dict[c]=='num']
    
    
    df_real_aux[cols_featuresNum]=df_real_aux[cols_featuresNum].astype(np.int)
    df_fake[cols_featuresNum]=df_fake[cols_featuresNum].astype(np.int)
    df_real_aux[cols_toOneHot]=df_real_aux[cols_toOneHot].astype(np.str)
    df_fake[cols_toOneHot]=df_fake[cols_toOneHot].astype(np.str)
    
    df_real_aux[col_target]=1
    df_fake[col_target]=0
      
    classifier='rf'
    df_final_filter=classifier_filter(df_real_aux,df_fake,col_target,cols_toOneHot,cols_featuresNum,classifier, size_test=0.3)
    
    

    
    
    #Cross evaluation: BL clasifier  
    df_real_aux=df_real.head(df_final.shape[0]).copy(deep=True)
    df_fake=df_final.copy(deep=True)
    
    
    col_target='BL'
    cols_toOneHot=[c for c in cols_dict if (cols_dict[c]=='cat' and c!=col_target)]
    cols_featuresNum=[c for c in cols_dict if cols_dict[c]=='num']
    
    
    df_real_aux[cols_featuresNum]=df_real_aux[cols_featuresNum].astype(np.int)
    df_fake[cols_featuresNum]=df_fake[cols_featuresNum].astype(np.int)
    df_real_aux[cols_toOneHot]=df_real_aux[cols_toOneHot].astype(np.str)
    df_fake[cols_toOneHot]=df_fake[cols_toOneHot].astype(np.str)
    
    df_real_aux[col_target]=df_real_aux[col_target].astype(np.str)
    df_fake[col_target]=df_fake[col_target].astype(np.str)
    
    
    classifier='rf'
    crossEvaluation(df_real_aux,df_fake,col_target,cols_toOneHot,cols_featuresNum,classifier, size_test=0.3)
    
    

    #Cross evaluation: Nationality    
    df_real_aux=df_real.head(df_final.shape[0]).copy(deep=True)
    df_fake=df_final.copy(deep=True)
    
    
    col_target='Nationality2'
    cols_toOneHot=[c for c in cols_dict if (cols_dict[c]=='cat' and c!=col_target)]
    cols_featuresNum=[c for c in cols_dict if cols_dict[c]=='num']
    
    
    df_real_aux[cols_featuresNum]=df_real_aux[cols_featuresNum].astype(np.int)
    df_fake[cols_featuresNum]=df_fake[cols_featuresNum].astype(np.int)
    df_real_aux[cols_toOneHot]=df_real_aux[cols_toOneHot].astype(np.str)
    df_fake[cols_toOneHot]=df_fake[cols_toOneHot].astype(np.str)
    
    df_real_aux[col_target]=df_real_aux[col_target].astype(np.str)
    df_fake[col_target]=df_fake[col_target].astype(np.str)
    
    
    classifier='rf'#logreg
    crossEvaluation(df_real_aux,df_fake,col_target,cols_toOneHot,cols_featuresNum,classifier, size_test=0.3)
    




############

#convert string format flag into tuple
def parseFlags_tuple(tuple_str):
    
    s=tuple_str[1:-1]
    t=s.split(',')
    
    return [int(d) for d in t]



class NoiseSampler(object):
    def __init__(self, z_dim,batch_size, z_normal):
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.z_normal = z_normal
    
    def __call__(self,*args):
        if len(args)==0:
            size=self.batch_size
        elif len(args)==1:
            size=args[0]
        else:
            raise Exception('Too many inputs')
            
            
        if self.z_normal:
            random_inputs = np.random.normal(loc=0.0, scale=1.0, size=[size, self.z_dim])        
        else:
            random_inputs = np.random.uniform(low=0.0, high=1.0, size=[size, self.z_dim])
                  
        return random_inputs



class Critic(object):
    def __init__(self, h):
        self.h = h

    def __call__(self, x, x_):
        _,h_x = self.h(x)
        _,h_x_ = self.h(x_)
        
        return tf.norm(h_x - h_x_, axis=1) - tf.norm(h_x, axis=1)



class CramerGAN(object):
    def __init__(self, x_dim, z_dim, g_net, d_net, x_sampler, z_sampler, scale=10.0, learning_rate=1e-4):
        self.d_net = d_net
        self.g_net = g_net
        self.critic = Critic(d_net)
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z1 = tf.placeholder(tf.float32, [None, self.z_dim], name='z1')
        self.z2 = tf.placeholder(tf.float32, [None, self.z_dim], name='z2')

        self.x1_,_ = self.g_net(self.z1, reuse=False)
        self.x2_,_ = self.g_net(self.z2)

        _,self.d_out  = d_net(self.x, reuse=False)

        self.g_loss = tf.reduce_mean(self.critic(self.x, self.x2_) - self.critic(self.x1_, self.x2_))
        self.d_loss = -self.g_loss

        # interpolate real and generated samples
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.x1_
        d_hat = self.critic(x_hat, self.x2_)

        ddx = tf.gradients(d_hat, x_hat)[0]
        print(ddx.get_shape().as_list())
        ddx = tf.norm(ddx, axis=1)
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)

        self.d_loss = self.d_loss + ddx

        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9)\
                .minimize(self.d_loss, var_list=self.d_net.vars)
            self.g_adam = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9)\
                .minimize(self.g_loss, var_list=self.g_net.vars)


        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        


    def train(self, header, dirname, batch_size, num_batches, interv, encoded_size, decoded_size):
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        print(num_batches)
        for t in range(0, num_batches):
            d_iters = FLAGS.n_critics
            for _ in range(0, d_iters):
                bx = self.x_sampler(batch_size)
                bz1 = self.z_sampler()
                bz2 = self.z_sampler()
                self.sess.run(self.d_adam, feed_dict={self.x: bx, self.z1: bz1, self.z2: bz2})

            bx = self.x_sampler(batch_size)
            bz1 = self.z_sampler()
            bz2 = self.z_sampler()
            self.sess.run(self.g_adam, feed_dict={self.z1: bz1, self.x: bx, self.z2: bz2})

            if t % FLAGS.evaluate_every == 0:
                bx = self.x_sampler(batch_size)
                bz1 = self.z_sampler()
                bz2 = self.z_sampler()
                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z1: bz1, self.z2: bz2}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z1: bz1, self.z2: bz2, self.x: bx}
                )
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                        (t, time.time() - start_time, d_loss, g_loss))

            if t % FLAGS.interv == 0 and t!=0:
                decodedSize = decoded_size
                    
                bz1 = self.z_sampler(decodedSize)
                bx= self.sess.run(self.x1_, feed_dict={self.z1: bz1})

                # subsample encoded datato avoid large files
                subn = encoded_size
                full = np.arange(bx.shape[0])
                choice = np.random.choice(full,subn)
                bx_sub = bx[choice,:]

                
                fake_d_out = self.sess.run(self.d_out, feed_dict={self.x: bx_sub})               
    
    
                real_encodedfnameout = dirname + "originalEncodedCat.csv"
                fake_encodedfnameout = dirname + "fakeEncodedCat.csv"
            
                fake_decodedfnameout = dirname + "fakeDecodedCat.csv"
                real_decodedfnameout = dirname + "originalDecodedCat.csv"

                header= ','.join(map(str,self.x_sampler.cenc.columns))
                np.savetxt(fake_encodedfnameout+str(t)+".gz",fake_d_out,delimiter=',',header=header, fmt='%.2e')

    
                if t % interv == 0:
                    dfout = self.x_sampler.cenc.inverse_transform(bx)
                    dfout.to_csv(fake_decodedfnameout+str(t)+".gz",  index=False , compression="gzip", float_format='%.2f')



            
                # real samples                
                realSubsample = self.x_sampler(decodedSize)
                # subsample encoded datato avoid large files
                full = np.arange(decodedSize)
                choice = np.random.choice(full,subn)
                realSubsample_sub = realSubsample[choice,:]

                real_d_out = self.sess.run(self.d_out, feed_dict={self.x: realSubsample_sub})

                np.savetxt(real_encodedfnameout+str(t)+".gz",real_d_out,delimiter=',',header=header, fmt='%.2e')
                
                if t % interv == 0:
                    dfdec = self.x_sampler.cenc.inverse_transform(realSubsample)
                    dfdec.to_csv(real_decodedfnameout+str(t)+".gz",  index=False , compression="gzip", float_format='%.2f')
    
    
    
    def sampler(self,n_batches):
        
        batches_res=[]
        for i in range(n_batches):
                  
            bz1 = self.z_sampler()
            bx= self.sess.run(self.x1_, feed_dict={self.z1: bz1})
            batches_res.append(np.squeeze(bx) )
  
    
        returnable=np.concatenate(batches_res,axis=0)
        dfout = self.x_sampler.cenc.inverse_transform(returnable)
        
        return dfout
     
        
    def saveModel(self,saver,filenameModel):
        saver.save(self.sess, filenameModel)
        
        
        
    def apply_h(self,dirname, ckpt_fname, input_fname, original_cat_cols,use_cols):
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt_fname)

        encoded_output_fName = os.path.join(dirname, os.path.basename(input_fname))+"_encodedH.csv"
        
        data_in=pd.read_csv(input_fname, usecols=use_cols)
                
        input_chunk = self.x_sampler.cenc.transform(data_in,original_cat_cols)
        input_d_out = self.sess.run(self.d_out, feed_dict={self.x: input_chunk})
        np.savetxt(encoded_output_fName,input_d_out,delimiter=',', fmt='%.2e')





##################

def main(_):
       
    FLAGS.genDims=parseFlags_tuple(FLAGS.genDims_str)
    FLAGS.discDims=parseFlags_tuple(FLAGS.discDims_str)
    
    
    #summary and other directories
    summaries_dir = os.path.join(FLAGS.traindir, 'summaries')
    plots_dir = os.path.join(FLAGS.traindir, 'plots')
    generated_data_dir = os.path.join(FLAGS.traindir, 'generated_data',FLAGS.outFile[:-4])
    try: 
        os.makedirs(FLAGS.traindir)
    except: 
        pass
    try: 
        os.makedirs(summaries_dir)
    except: 
        pass
    try: 
        os.makedirs(plots_dir)
    except: 
        pass
    try: 
        os.makedirs(generated_data_dir)
    except: 
        pass


 

    filenameIn='./data/train.csv' 
    #define datatype of each column
    cols_dict={'Origin_Country':'cat',
               'Destination_Country':'cat',
               'OfficeIdCountry':'cat',
               'SaturdayStay':'cat',
               'PurchaseAnticipation':'num',
               'NumberPassenger':'cat',
               'StayDuration':'num',
               'Title':'cat',
               'TravelingWithChild':'cat',
               'ageAtPnrCreation':'num',
               'Nationality2':'cat',
               'BL':'cat'}
    use_cols = [item for item in cols_dict]
    cat_cols = [item for item in cols_dict if cols_dict[item]=='cat']
    
    #n points to genreate for intermediate results
    n_points=12000

    
  
    x_sampler = Data(file = filenameIn ,cat_cols=cat_cols,
                use_cols = use_cols)
    print("Training data shape: "+ str(x_sampler.shape))
    x_dim = x_sampler.shape[1]
    x_size=x_sampler.shape[0]
    num_batches=int(FLAGS.max_epoch *x_size/FLAGS.batch_size)
    z_sampler = NoiseSampler(FLAGS.zDim, FLAGS.batch_size, FLAGS.z_normal)
    catEmbeddingDims = [min(max(int(round(FLAGS.k*np.log(v_size) ) ),2),v_size) for v_size in x_sampler.cenc.cat_card]

    
    #create model's graph
    d_net = model.GenDisc(disc=True, dims=FLAGS.discDims, out_dims=FLAGS.discriminatorOut_dims, dims_nCross=FLAGS.discDims_nCross, 
                          catEmbeddingDims = catEmbeddingDims, ohFeatureIndices = x_sampler.cenc.feature_indices,numColsLen = len(x_sampler.cenc.num_cols))    
    g_net = model.GenDisc(disc=False, dims=FLAGS.genDims, out_dims=x_dim,dims_nCross=FLAGS.discDims_nCross,
                          catEmbeddingDims = catEmbeddingDims,ohFeatureIndices = x_sampler.cenc.feature_indices,numColsLen = len(x_sampler.cenc.num_cols)) 
    cgan = CramerGAN(x_dim=x_dim, z_dim=FLAGS.zDim, g_net=g_net, d_net=d_net, x_sampler=x_sampler, z_sampler=z_sampler)



    #train model
    cgan.train(header=use_cols, dirname=generated_data_dir, batch_size=FLAGS.batch_size, num_batches=num_batches,
               interv=FLAGS.interv, decoded_size=n_points, encoded_size=n_points)
    
    
    #save trained model
    filenameModel = os.path.join(generated_data_dir, "model.ckpt")
    saver = tf.train.Saver()   
    cgan.saveModel(saver,filenameModel)
    
    
    #generate synthetic data
    n_finalGen=x_size
    dfout=cgan.sampler(int(n_finalGen/FLAGS.batch_size))
    filename = os.path.join(generated_data_dir, FLAGS.outFile)
    dfout.to_csv(filename,  index=False , float_format='%.2f')


      
    #use trained model to transform data using the h fucntion of the critic
    #used for evaluation
    ckpt_fname=os.path.join(generated_data_dir, "model.ckpt")
    dirname_out=generated_data_dir
    input_fname='./data/test.csv'
    cgan.apply_h(dirname_out, ckpt_fname, input_fname, cat_cols,use_cols)
    
    input_fname=filenameIn
    cgan.apply_h(dirname_out, ckpt_fname, input_fname, cat_cols,use_cols)

    input_fname=os.path.join(generated_data_dir, FLAGS.outFile)
    cgan.apply_h(dirname_out, ckpt_fname, input_fname, cat_cols,use_cols)



###############


if __name__ == '__main__':
    tf.app.run()
    


   
    
    
    
    
    
    