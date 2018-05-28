import os
import numpy as np
import tensorflow as tf

from collections import OrderedDict
from caffe_builder_utils import *

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
        
        print("key",key,val)
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
    online_layer_dict = OrderedDict()
    with tf.variable_scope("q_networks/online") as scope:
        for n,layer_info in enumerate(CNN_struct["layer"]):
            try:
                build_layer(layer_info,online_layer_dict)
            except Exception as e:
                raise Exception("Error during layer %d creation: %s"%(n+1,str(e)))
        online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
        online_vars = {var.name[len(scope.name):]: var
                                  for var in online_vars}
    print(online_vars)
    target_layer_dict = OrderedDict()
    with tf.variable_scope("q_networks/target") as scope:
        for n,layer_info in enumerate(CNN_struct["layer"]):
            try:
                build_layer(layer_info,target_layer_dict)
            except Exception as e:
                raise Exception("Error during layer %d creation: %s"%(n+1,str(e)))
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
        target_vars = {var.name[len(scope.name):]: var
                                  for var in target_vars}
    print(target_vars)
    copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
    copy_online_to_target = tf.group(*copy_ops)
    if "input" in online_layer_dict:
        input_layer = online_layer_dict["input"]["tf_obj"]
    else:
        input_layer   = list(online_layer_dict.items())[0][1]["tf_obj"]
    if "output" in online_layer_dict:
        input_layer = online_layer_dict["output"]["tf_obj"]
    else:
        output_layer  = list(online_layer_dict.items())[-1][1]["tf_obj"]
    if "readout" in online_layer_dict:
        readout_layer = online_layer_dict["readout"]["tf_obj"]
    else:
        readout_layer = list(online_layer_dict.items())[-1][1]["tf_obj"]
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
    build_funcs["reshape"]      = build_reshape
    build_funcs["dense"]        = build_dense


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