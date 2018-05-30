import tensorflow as tf

def get_padding_type(kernel_params, input_shape, output_shape):
    '''
        Tensorflow only supports "SAME" and "VALID" Padding
    '''
    
    [i_height,i_width] = input_shape
    i_width = input_shape[1]
    o_height
    k_h, k_w, s_h, s_w, p_h, p_w = kernel_params
    s_o_h = np.ceil(i_height / float(s_h))
    s_o_w = np.ceil(i_width / float(s_w))
    if (output_shape.height == s_o_h) and (output_shape.width == s_o_w):
        return 'SAME'
    v_o_h = np.ceil((i_height - k_h + 1.0) / float(s_h))
    v_o_w = np.ceil((i_width - k_w + 1.0) / float(s_w))
    if (output_shape.height == v_o_h) and (output_shape.width == v_o_w):
        return 'VALID'
    return None
def build_var(var_type,value,shape=[1,1,1,1]):
    recognized_types = {}
    recognized_types["constant"]= tf.constant_initializer
    recognized_types["truncated_normal"]  = tf.truncated_normal
    recognized_types["gaussian"] = recognized_types["truncated_normal"]
    recognized_types["variance_scaling"] = tf.variance_scaling_initializer
    
    if var_type.lower() == "xavier":
        print("Warning, weight filler type \"xavier\" is not supported")
    if not var_type in recognized_types:
        raise Exception("Type %s is not a recognized variable type"%(var_type))
    #recognized_types["constant"]=tf.Constant
    #TODO: Figure out how to pass args into the variable generator
    if var_type == "gaussian" or var_type == "truncated_normal":
        return tf.Variable(recognized_types[var_type](shape=shape,stddev=value))
    elif var_type == "variance_scaling":
        return recognized_types["variance_scaling"]()
    else:
        return tf.Variable(recognized_types[var_type](value))
def build_input(parent,layer_info):
    out_shape = fix_shape(layer_info["input_param"]["shape"]["dim"])
    return tf.placeholder("float", out_shape)

def build_convolution(parent,layer_info):
    #Assign Default Values
    #num_output     = 0            #Required input
    #kernel_size    = [5,5]    #Required input
    layer_pad   = "SAME"
    stride      = [1,1,1,1]
    bias_term   = False
    activation=tf.nn.relu
    initializer = tf.variance_scaling_initializer()

    #For weight_filler
    w_type = "constant"
    w_val = 0
    b_type = "constant"
    b_val = 0

    #Params are currently ignored
    if "param" in layer_info:
        params = layer_info["param"]
        print("Warning: param arg in layer %s is ignored"%(layer_info["name"]))

    #convolution parameter handling
    layer_conv_info = layer_info["convolution_param"]
    if not "num_output" in layer_conv_info:
        raise Exception("Layer %s missing required num_output arg, "%(layer_info["name"]))


    num_output = layer_conv_info["num_output"]
    if "kernel_size" in layer_conv_info:
        kernel_size = fix_shape(layer_conv_info["kernel_size"]["dim"])
    elif "kernel_h" in layer_conv_info:
        kernel_size = fix_shape([layer_conv_info["kernel_h"],layer_conv_info["kernel_w"]])
    else:
        raise Exception("Layer %s missing required kernel_size arg, "%(layer_info["name"]))
    if not isinstance(stride, list) and len(stride) == 4:
        raise Exception("Error, invalid value for kernel_size %s in layer for Layer %s"%(str(kernel_size),layer_info["name"]))  


    if "weight_filler" in layer_conv_info:
        if "type" in layer_conv_info["weight_filler"]:
            w_type = layer_conv_info["weight_filler"]["type"]
        if "value" in layer_conv_info["weight_filler"]:
            w_val = layer_conv_info["weight_filler"]["value"]
    if "bias_filler" in layer_conv_info:
        if "type" in layer_conv_info["bias_filler"]:
            b_type = layer_conv_info["bias_filler"]["type"]
        if "value" in layer_conv_info["bias_filler"]:
            b_val = layer_conv_info["bias_filler"]["value"]


    if "bias_term" in layer_conv_info:
        if layer_conv_info["bias_term"].lower() == "true":
            bias_term = True
        elif layer_conv_info["bias_term"].lower() == "false":
            bias_term = False
        else:
            raise Exception("Unrecognized value %s for bias_term in Layer %s"%(layer_conv_info["bias_term"],layer_info["name"]))


    if "pad" in layer_conv_info:
        layer_pad = layer_conv_info["pad"]
    elif "pad_h" in layer_conv_info:
        layer_pad = [layer_conv_info["pad_h"],layer_conv_info["pad_w"]]
    if not str(layer_pad).upper() == "SAME" and not str(layer_pad).upper() == "VALID":
        raise Exception("Error, Tensorflow only supports \"same\" and \"valid\" padding for Layer %s"%(layer_info["name"]))  


    if "stride" in layer_conv_info:
        if isinstance(layer_conv_info["stride"]["dim"],list):
            stride = fix_shape(layer_conv_info["stride"]["dim"])
        else:
            stride = layer_conv_info["stride"]["dim"]
    elif "stride_h" in layer_conv_info:
        stride = fix_shape([layer_conv_info["stride_h"],layer_conv_info["stride_w"]])
    #if not isinstance(stride, list) and len(stride) == 4:
    #    raise Exception("Error, invalid value for stride %s in layer for Layer %s"%(str(stride),layer_info["name"]))  

    if "group" in layer_conv_info:
        group = layer_conv_info["group"]
        print("Warning, group parameter in Layer %s is ignored"%(layer_info["name"]))
    if "activation" in layer_conv_info:
        activation_type = layer_conv_info["activation"].lower()
        if activation_type == "relu":
            activation = tf.nn.relu
        else:
            raise Exception("Unrecognized activation type %s in layer %s"%(activation_type,layer_info["name"]))

            
    #TODO: Generate the nn args properly
    #print("BUILDING W")
    W_var = build_var(w_type,w_val)
    B_var = None
    #if bias_term:
    #    B_var = build_var(b_type,b_val)
    #else:
    #    B_var = None
    print("Building Convolutional layer with following parameters: " + 
        str("num_filters %s, kernel_size %s, padding %s, stride %s, use_bias %s"%(
            num_output,str(kernel_size),str(layer_pad.lower()),str(stride),bias_term)))

    return tf.layers.conv2d(parent, filters=num_output,kernel_size=kernel_size, 
        padding=layer_pad.upper(), strides = stride,activation=activation,kernel_initializer=initializer)
    
def build_pooling(parent,layer_info):

    #default args
    #kernel_size = [1,5,5,1] #Required input
    layer_pad = "SAME"
    stride = [1,1,1,1]

    valid_pool_types = {}
    valid_pool_types["MAX"] = tf.nn.max_pool
    valid_pool_types["AVE"] = tf.nn.avg_pool


    layer_pool_info = layer_info["pooling_param"]
    pool = "MAX"

    if "kernel_size" in layer_pool_info:
        kernel_size = fix_shape(layer_pool_info["kernel_size"]["dim"])
    elif "kernel_h" in layer_pool_info:
        kernel_size = fix_shape([1,layer_pool_info["kernel_h"],layer_pool_info["kernel_w"],1])
    else:
        raise Exception("Layer %s missing required kernel_size arg, "%(layer_info["name"]))

    if "pad" in layer_pool_info:
        layer_pad = layer_pool_info["pad"]
    elif "pad_h" in layer_pool_info:
        raise Exception("pad_h,pad_w not supported")
    if not str(layer_pad).upper() == "SAME" and not str(layer_pad).upper() == "VALID":
        raise Exception("Error, Tensorflow only supports \"same\" and \"valid\" padding for Layer %s"%(layer_info["name"]))  

    if "stride" in layer_pool_info:
        stride = fix_shape(layer_pool_info["stride"]["dim"])
    elif "stride_h" in layer_pool_info:
        stride = fix_shape([1,layer_pool_info["stride_h"],layer_pool_info["stride_w"],1])
    if not isinstance(stride, list) and len(stride) == 4:
        raise Exception("Error, invalid value for stride %s in layer for Layer %s"%(str(stride),layer_info["name"]))  


    if "pool" in layer_pool_info:
        pool = layer_pool_info["pool"].upper()

    if not pool in valid_pool_types:
        if pool == "STOCHASTIC":
            print("Stochastic pooling is not supported")
        raise Exception("Unrecognized pool type %s in layer %s"%(layer_info["type"],layer_info["name"]))



    #tf.nn.max_pool(parent, strides = strides,ksize = ksize, padding = "SAME")
    return valid_pool_types[pool](parent,strides=stride,ksize=kernel_size,padding=layer_pad)

def build_innerproduct(parent,layer_info):
    layer_ip_info = layer_info["inner_product_param"]
    if not "num_output" in layer_ip_info:
        raise Exception("Layer %s missing required num_output arg, "%(layer_info["name"]))

    if "weight_filler" in layer_ip_info:
        w_type = "constant"
        w_val = 0
        if "type" in layer_ip_info["weight_filler"]:
            w_type = layer_ip_info["weight_filler"]["type"]
        if "value" in layer_ip_info["weight_filler"]:
            w_type = layer_ip_info["weight_filler"]["value"]

    if "bias_filler" in layer_ip_info:
        b_type = "constant"
        b_val = 0
        if "type" in layer_ip_info["bias_filler"]:
            b_type = layer_ip_info["bias_filler"]["type"]
        if "value" in layer_ip_info["bias_filler"]:
            b_type = layer_ip_info["bias_filler"]["value"]

    if "bias_term" in layer_ip_info:
        bias_term = layer_ip_info["bias_filler"]["bias_term"]

    #TODO: Do these inputs properly
    return tf.reduce_sum(tf.multiply(parent,parent),1,keep_dims=True)

def build_relu(parent,layer_info):
    negative_slope=0
    if "relu_param" in layer_info:
        if "negative_slope" in layer_info["relu_param"]:
            negative_slope = 0
    
    #TODO: Verify that this is the correct way to call leaky relu
    if negative_slope != 0:
        return tf.nn.relu(parent)
    else:
        return tf.nn.leaky_relu(parent,negative_slope)
def build_softmax(parent,layer_info):
    return tf.nn.softmax(parent)
def build_flatten(parent,layer_info):
    return tf.layers.flatten(parent)
def build_reshape(parent,layer_info):
    shape = [None,5]   

    if "shape" in layer_info:
        shape = fix_shape(layer_info["shape"]["dim"])
    else:
        raise Exception("Missing shape arguement in layer %s"%(layer_info["name"]))

    return tf.reshape(parent,shape)
    
def build_dense(parent,layer_info):
    activation = None
    kernel_init = tf.variance_scaling_initializer()
    activation_type = "None"
    
    if not "dense_param" in layer_info:
        raise Exception("Missing dense_param in layer %s"%(layer_info["name"]))
    
    layer_dense = layer_info["dense_param"]
    if "num_output" in layer_dense:
        num_output = layer_dense["num_output"]
    else:
        raise Exception("Missing num_output arguement in layer %s"%(layer_info["name"]))
    if "activation" in layer_dense:
        activation_type = layer_dense["activation"]
        if activation_type == "relu":
            activation = tf.nn.relu
    print("Building dense layer with num_output %d, initializer %s and activation %s"%(num_output,activation,kernel_init))
    if activation is None:
        return tf.layers.dense(parent,num_output,kernel_initializer=kernel_init)
    else:
        return tf.layers.dense(parent,num_output,activation=activation,kernel_initializer=kernel_init)
def fix_shape(shape_in):
    shape_out = []
    for val in shape_in:
        try:
            val = int(val)
            shape_out.append(val)
        except Exception as e:
            if val.lower() == "none":
                shape_out.append(None)
            else:
                raise Exception("Unrecognized value in shape %s"%(val))
    return shape_out