import os
import numpy as np
import tensorflow as tf

from collections import OrderedDict

def fix_shape(shape_in):
    shape_out = []
    for val in shape_in:
        if val < 0:
            shape_out.append(None)
        else:
            shape_out.append(val)
    return shape_out
def segment_prototxt(lines,lineid,struct):
    
    while True:
        if lineid >= len(lines):
            return struct,lineid
        cur_line = lines[lineid]
        lineid += 1
        #print("LINE ",cur_line)
        if "{" in cur_line:
            split_line = cur_line.split("{")
            key = split_line[0].replace(" ","").replace(":","")
            val,lineid = segment_prototxt(lines,lineid,OrderedDict())
            #print("KEY %s"%(str(key)))
            #print("VAL %s"%(str(val)))

        elif "}" in cur_line:
            #print("RETURNING VAL %s"%struct)
            return struct,lineid
        elif ":" in cur_line:
            split_line = cur_line.split(":")
            key = split_line[0].replace(" ","")
            val = split_line[1].replace(" ","")
            try:
                val = int(val)
            except:
                val = val.replace("\"","").replace("\'","")
                pass

        if key in struct:
            if isinstance(struct[key],list):
                struct[key].append(val)
            else:
                struct[key] = [struct[key],val]
        else:
            struct[key] = val

def parse_prototxt(filepath):
    '''
        Parsing makes the following assumptions (See Example.prototxt)

        - every line has at most 1 key-value pair
        - An object with multiple values (indicated by brackets), will have
            a newline after the opening brackets
        - Close brackets have their own line
        - Strings must have no white space
        - No assumptions are made about white space
        - Assumes all layers are defined after their inputs are defined
    '''
    if(not os.path.isfile(filepath)):
        raise Exception("Prototxt not found: %s"%(filepath))

    #proto_data = "name: \"LeNet\"\nlayer {\n  name: \"data\"\n  type: \"Input\"\n  top: \"data\"\n  input_param { shape: { dim: 64 dim: 1 dim: 28 dim: 28 } }\n}"
    with open(filepath) as file:
        proto_data = file.read()
    proto_data = proto_data.split("\n")
    CNN_struct = OrderedDict()
    segment_prototxt(proto_data,0,CNN_struct)
    def print_dict(cur_dict,tabs_str=" "):
        cur_str = ""
        for key in cur_dict:
            val_n = cur_dict[key]
            if not isinstance(val_n,list):
                val_n = [val_n]
            for val in val_n:
                if isinstance(val,dict):
                    val_str = print_dict(val,tabs_str+" ")
                    cur_str += str("%s%s :\n"%(tabs_str,key))
                    for line in val_str.split("\n"):
                        cur_str += str("%s%s\n"%(tabs_str+" ",line))
                else:
                    val_str = str(val)
                    cur_str += str("%s%s = %s\n"%(tabs_str,key,val_str))
        return cur_str
    #print(print_dict(CNN_struct,""))
    return CNN_struct

def build_CNN(caffe_file=None,CNN_struct = None):
    #This will build the tensorflow implementation of the specified network

    if not CNN_struct is None:
        pass
    elif not caffe_file is None:
        try:
            CNN_struct = parse_prototxt(caffe_file)
        except Exception as e:
            raise Exception("Error during .prototxt file parsing %s"%str(e))
    else:
        raise Exception("build_CNN, either caffe_file, cor CNN_struct must be specified")

    if "name" in CNN_struct:
        CNN_name = CNN_struct["name"]
    else:
        CNN_name = "name_unspecified"
    layer_dict = OrderedDict()
    for n,layer_info in enumerate(CNN_struct["layer"]):
        try:
            build_layer(layer_info,layer_dict)
        except Exception as e:
            raise Exception("Error during layer %d creation: %s"%(n+1,str(e)))
            
    if "input" in layer_dict:
        input_layer = layer_dict["input"]["tf_obj"]
    else:
        input_layer   = layer_dict.items()[0][1]["tf_obj"]
    if "output" in layer_dict:
        input_layer = layer_dict["output"]["tf_obj"]
    else:
        output_layer  = layer_dict.items()[-1][1]["tf_obj"]
    if "readout" in layer_dict:
        readout_layer = layer_dict["readout"]["tf_obj"]
    else:
        readout_layer = layer_dict.items()[-1][1]["tf_obj"]

    return CNN_name,input_layer,output_layer,readout_layer

def build_layer(layer_info,layer_dict):
    '''
        layer_info: is a dictionary object that contains the .prototxt style information
        layer_dict: is a dictionary object that contains all existing layers, keyed by the layer "top"
    '''
    build_funcs = {}
    build_funcs["input"]        = build_input
    build_funcs["convolution"]  = build_convolution
    build_funcs["pooling"]      = build_pooling
    build_funcs["innerproduct"] = build_innerproduct
    build_funcs["relu"]         = build_relu
    build_funcs["softmax"]      = build_softmax
    build_funcs["flatten"]      = build_flatten


    layer = OrderedDict()
    layer_name = layer_info["name"]
    layer_type = layer_info["type"].lower()
    layer_top  = layer_info["top"]

    if layer_top in layer_dict:
        existing_layer_name = layer_dict[layer_top]["name"]
        raise Exception("Error: Layer %s has top %s which already exists in layer %s"%(layer_name,layer_top,existing_layer_name))

    #All layers but input layer will have a bottom from which input is taken
    parent_layer = None
    if not layer_type.lower() == "input":
        layer_bottom    = layer_info["bottom"]
        layer["bottom"] = layer_bottom
        parent_obj      = layer_dict[layer_bottom]
        parent_layer = parent_obj["tf_obj"]
        
    layer["name"] = layer_name
    layer["type"] = layer_type
    layer["top"]  = layer_top
    layer_dict[layer_top] = layer
    #This will handle each of the different layer types
    if not layer_type in build_funcs:
        raise Exception("Unrecognized layer type %s and name %s"%(layer_info["type"],layer_name))
    

    layer["tf_obj"] = build_funcs[layer_type](parent_layer,layer_info)

def build_var(var_type,value,shape=[1,1,1,1]):
    recognized_types = {}
    recognized_types["constant"]= tf.constant_initializer
    recognized_types["truncated_normal"]  = tf.truncated_normal
    recognized_types["gaussian"] = recognized_types["truncated_normal"]

    if var_type.lower() == "xavier":
        print("Warning, weight filler type \"xavier\" is not supported")
    if not var_type in recognized_types:
        raise Exception("Type %s is not a recognized variable type"%(var_type))
    #recognized_types["constant"]=tf.Constant
    #TODO: Figure out how to pass args into the variable generator
    if var_type == "gaussian" or var_type == "truncated_normal":
        return tf.Variable(recognized_types[var_type](shape=shape,stddev=value))
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
    bias_term   = True

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
        stride = fix_shape(layer_conv_info["stride"]["dim"])
    elif "stride_h" in layer_conv_info:
        stride = fix_shape([layer_conv_info["stride_h"],layer_conv_info["stride_w"]])
    if not isinstance(stride, list) and len(stride) == 4:
        raise Exception("Error, invalid value for stride %s in layer for Layer %s"%(str(stride),layer_info["name"]))  

    if "group" in layer_conv_info:
        group = layer_conv_info["group"]
        print("Warning, group parameter in Layer %s is ignored"%(layer_info["name"]))
            
    #TODO: Generate the nn args properly
    #print("BUILDING W")
    #_var = build_var(w_type,w_val)
    B_var = None
    #if bias_term:
    #    B_var = build_var(b_type,b_val)
    #else:
    #    B_var = None

    print("Building Convolutional layer with following parameters: " + 
        str("num_filters %s, kernel_size %s, padding %s, stride %s, use_bias %s"%(
            num_output,str(kernel_size),str(layer_pad.lower()),str(stride),bias_term)))

    return tf.layers.conv2d(parent, filters=num_output,kernel_size=kernel_size, 
        padding=layer_pad.lower(), strides = stride,use_bias=bias_term,bias_initializer=B_var)
    
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