import numpy as np

def activation_function(input, activation_name):
    """
    Returns: [output, derivative], each rounded to 4 decimal places.
    """

    input = float(input)
    alpha_leaky_relu = 0.01 
    if activation_name == "relu":
        output =  max(0.0, input)
        derivative = 1.0 if input > 0 else 0.0
    elif activation_name == "sigmoid":
        sigmoid = 1.0 / ( 1 + np.exp(-input)) # Calculator sigmod fomulation
        output = float(sigmoid)
        derivative = sigmoid * (1-sigmoid)
    elif activation_name == "leaky_relu":
        output = input if input > 0.0 else alpha_leaky_relu * input
        derivative = 1.0 if input > 0.0 else alpha_leaky_relu
    elif activation_name == "tanh":
        tanh = np.tanh(input)
        output = float(tanh)
        derivative = 1 - tanh**2
    elif activation_name == "gelu":
        c  = np.sqrt(2.0 /  np.pi )
        inner_tanh = c * (input + 0.044715 * input**3)
        tanh = float(np.tanh(inner_tanh))
        cal_gelu = 0.5 * input * (1 + tanh)
        output = float(cal_gelu)
        inner_deriv = c * (1 + 3 * 0.044715 * input ** 2)
        sech2 = 1 - tanh ** 2
        derivative = float(0.5 * (1 + tanh) + 0.5 * input * sech2 * inner_deriv)
    elif activation_name == "swish":
        s = 1.0 / (1.0 + np.exp(-input))
        output = float(input * s)
        derivative = float(s + input * s * (1 - s))
    
    return [round(output,4), round(derivative,4)]

