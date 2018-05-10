import os
import numpy as np
import tensorflow as tf

from collections import OrderedDict


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

    try:
        CNN_name = CNN_struct["name"]
        layer_dict = OrderedDict()
        for layer_info in CNN_struct["layer"]:
            build_layer(layer_info,layer_dict)
    except Exception as e:
        raise Exception("Error during layer creation: %s"%str(e))


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


    layer = OrderedDict()
    layer_name = layer_info["name"]
    layer_type = layer_info["type"].lower()
    layer_top  = layer_info["top"]

    #All layers but input layer will have a bottom from which input is taken
    parent_obj = None
    if not layer_type.lower() == "input":
        layer_bottom    = layer_info["bottom"]
        layer["bottom"] = layer_bottom
        parent_obj      = layer_dict[layer_bottom]
        
    layer["name"] = layer_name
    layer["type"] = layer_type
    layer["top"]  = layer_top
    layer_dict[layer_top] = layer
    #This will handle each of the different layer types
    if not layer_type in build_funcs:
        raise Exception("Unrecognized layer type %s and name %s"%(layer_info["type"],layer_name))
    
    layer["tf_obj"] = build_funcs[layer_type](parent_obj,layer_info)

def build_var(var_type,value):
    recognized_types = {}
    recognized_types["constant"]= tf.Constant
    recognized_types["truncated_normal"]  = tf.truncated_normal
    #recognized_types["constant"]=tf.Constant
    #TODO: Figure out how to pass args into the variable generator

    return tf.Variable(recognized_types[var_type](value))

def build_input(parent,layer_info):
    out_shape = layer_info["input_param"]["shape"]["dim"]
    return tf.placeholder("float", out_shape)

def build_convolution(parent,layer_info):
    params = layer_info["param"]

    #convolution parameter handling
    layer_conv_info = layer_info["convolution_param"]
    if not "num_output" in layer_conv_info:
        raise Exception("Layer %s missing required num_output arg, "%(layer_info["name"]))

    num_output = layer_conv_info["num_output"]
    if "kernel_size" in layer_conv_info:
        kernel_size = layer_conv_info["kernel_size"]
    elif "kernel_h" in layer_conv_info:
        kernel_size = [layer_conv_info["kernel_h"],layer_conv_info["kernel_w"]]
    else:
        raise Exception("Layer %s missing required kernel_size arg, "%(layer_info["name"]))
    
    if "pad" in layer_conv_info:
        layer_pad = layer_conv_info["pad"]
    elif "pad_h" in layer_conv_info:
        layer_pad = [layer_conv_info["pad_h"],layer_conv_info["pad_w"]]

    if "stride" in layer_conv_info:
        stride = layer_conv_info["stride"]
    elif "stride_h" in layer_conv_info:
        stride = [layer_conv_info["stride_h"],layer_conv_info["stride_w"]]

    if "group" in layer_conv_info:
        group = layer_conv_info["group"]

    if "weight_filler" in layer_conv_info:
        w_type = "constant"
        w_val = 0
        if "type" in layer_conv_info["weight_filler"]:
            w_type = layer_conv_info["weight_filler"]["type"]
        if "value" in layer_conv_info["weight_filler"]:
            w_type = layer_conv_info["weight_filler"]["value"]

    if "bias_filler" in layer_conv_info:
        b_type = "constant"
        b_val = 0
        if "type" in layer_conv_info["bias_filler"]:
            b_type = layer_conv_info["bias_filler"]["type"]
        if "value" in layer_conv_info["bias_filler"]:
            b_type = layer_conv_info["bias_filler"]["value"]
            
    #TODO: Generate the nn args properly
    W_var = build_var(w_type,w_val)
    B_var = build_var(b_type,b_val)
    return tf.nn.conv2d(parent, W_var, strides = stride)
    
def build_pooling(parent,layer_info):
    valid_pool_types = {}
    valid_pool_types["max"] = tf.nn.max_pool
    valid_pool_types["ave"] = tf.nn.avg_pool


    layer_pool_info = layer_info["pooling_param"]
    pool = "max"

    if "kernel_size" in layer_pool_info:
        kernel_size = layer_pool_info["kernel_size"]
    elif "kernel_h" in layer_pool_info:
        kernel_size = [layer_pool_info["kernel_h"],layer_pool_info["kernel_w"]]
    else:
        raise Exception("Layer %s missing required kernel_size arg, "%(layer_info["name"]))

    if "pad" in layer_pool_info:
        layer_pad = layer_pool_info["pad"]
    elif "pad_h" in layer_pool_info:
        layer_pad = [layer_pool_info["pad_h"],layer_pool_info["pad_w"]]

    if "stride" in layer_pool_info:
        stride = layer_pool_info["stride"]
    elif "stride_h" in layer_pool_info:
        stride = [layer_pool_info["stride_h"],layer_pool_info["stride_w"]]

    if "pool" in layer_pool_info:
        pool = layer_pool_info["pool"].lower()

    if not pool in valid_pool_types:
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