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