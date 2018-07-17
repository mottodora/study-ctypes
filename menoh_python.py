# IPython log file
from __future__ import print_function

from ctypes import *
class MenohModelDataHandle(Structure):
    pass
menoh = CDLL('build/menoh/libmenoh.dylib')
model_data_handle = pointer(MenohModelDataHandle())
ret = menoh.menoh_make_model_data_from_onnx('data/VGG16.onnx', byref(model_data_handle))
print(ret)
class MenohVariableProfileTableBuilderHandle(Structure):
    pass
profile_table_builder_handle = pointer(MenohVariableProfileTableBuilderHandle())
ret = menoh.menoh_make_variable_profile_table_builder(byref(profile_table_builder_handle))
print(ret)

CONV1_1_IN_NAME = "140326425860192"
ret
func = menoh.menoh_variable_profile_table_builder_add_input_profile_dims_4
func.argtypes = (POINTER(MenohVariableProfileTableBuilderHandle), c_char_p, c_int, c_int, c_int, c_int, c_int)
ret = menoh.menoh_variable_profile_table_builder_add_input_profile_dims_4(profile_table_builder_handle, CONV1_1_IN_NAME, 0, 1, 3, 224, 224)
print(ret)
if ret != 0:
    get_error = menoh.menoh_get_last_error_message
    # get_error.restype = POINTER(c_char)
    get_error.restype = c_char_p
    data = get_error()
    print(data)
else:
    print("success")

FC6_OUT_NAME = "140326200777584"
ret = menoh.menoh_variable_profile_table_builder_add_output_profile(profile_table_builder_handle, FC6_OUT_NAME, 0)
print(ret)
#
SOFTMAX_OUT_NAME = "140326200803680"
ret = menoh.menoh_variable_profile_table_builder_add_output_profile(profile_table_builder_handle, SOFTMAX_OUT_NAME, 0)
print(ret)
class MenohVariableProfileTable(Structure):
    pass
profile_table = pointer(MenohVariableProfileTable())
func2 = menoh.menoh_build_variable_profile_table
func.argtypes = (MenohVariableProfileTableBuilderHandle, MenohModelDataHandle, POINTER(MenohVariableProfileTable))
# ret = menoh.menoh_build_variable_profile_table(profile_table_builder_handle, model_data_handle, byref(profile_table))
ret = func2(profile_table_builder_handle, model_data_handle, byref(profile_table))
print(ret)
if ret != 0:
    get_error = menoh.menoh_get_last_error_message
    # get_error.restype = POINTER(c_char)
    get_error.restype = c_char_p
    data = get_error()
    print(data)

softmax_out_dim_0 = c_int()
ret = menoh.menoh_variable_profile_table_get_dims_at(profile_table, SOFTMAX_OUT_NAME, 0, byref(softmax_out_dim_0))
print(ret)
softmax_out_dim_1 = c_int()
ret = menoh.menoh_variable_profile_table_get_dims_at(profile_table, SOFTMAX_OUT_NAME, 1, byref(softmax_out_dim_1))
print(ret)

ret = menoh.menoh_model_data_optimize(model_data_handle, profile_table)
print(ret)
class MenohModelBuilder(Structure):
    pass
model_builder = pointer(MenohModelBuilder())
ret = menoh.menoh_make_model_builder(profile_table, byref(model_builder))
print(ret)
input_buff = pointer((c_float*(1 * 3 * 224 * 224))())
ret = menoh.menoh_model_builder_attach_external_buffer(model_builder, CONV1_1_IN_NAME, input_buff)
print(ret)

class MenohModel(Structure):
    pass

model = pointer(MenohModel())
ret = menoh.menoh_build_model(model_builder, model_data_handle, "mkldnn", "", byref(model))
print(ret)
ret = menoh.menoh_delete_model_data(model_data_handle)
print(ret) # no error check?

fc6_output_buff = pointer((c_float*10)())
ret = menoh.menoh_model_get_variable_buffer_handle(model, FC6_OUT_NAME, byref(fc6_output_buff))
print(ret)

softmax_output_buff = pointer((c_float * softmax_out_dim_0.value * softmax_out_dim_1.value)())
ret = menoh.menoh_model_get_variable_buffer_handle(model, SOFTMAX_OUT_NAME, byref(softmax_output_buff))
print(ret)

for i in range(1 * 3 * 224 * 224):
    input_buff.contents[i] = 0.5

ret = menoh.menoh_model_run(model)
print(ret)

for i in range(10):
    print(fc6_output_buff.contents[i], end=" ")
print("")

# for i in range(softmax_out_dim_0.value):
#     for j in range(softmax_out_dim_1.value):
#         print(softmax_output_buff[i * softmax_out_dim_1.value + j].values, end=" ")
#     print("")

ret = menoh.menoh_delete_model(model)
print(ret)
ret = menoh.menoh_delete_model_builder(model_builder)
print(ret)
ret = menoh.menoh_delete_variable_profile_table_builder(profile_table_builder_handle)
print(ret)
