import ctypes


menoh = ctypes.CDLL('build/menoh/libmenoh.dylib')
menoh_error_list = [
    "menoh_error_code_success",
    "menoh_error_code_std_error",
    "menoh_error_code_unknown_error",
    "menoh_error_code_invalid_filename",
    "menoh_error_code_unsupported_onnx_opset_version",
    "menoh_error_code_onnx_parse_error",
    "menoh_error_code_invalid_dtype",
    "menoh_error_code_invalid_attribute_type",
    "menoh_error_code_unsupported_operator_attribute",
    "menoh_error_code_dimension_mismatch",
    "menoh_error_code_variable_not_found",
    "menoh_error_code_index_out_of_range",
    "menoh_error_code_json_parse_error",
    "menoh_error_code_invalid_backend_name",
    "menoh_error_code_unsupported_operator",
    "menoh_error_code_failed_to_configure_operator",
    "menoh_error_code_backend_error",
    "menoh_error_code_same_named_variable_already_exist",
]


class Handle(ctypes.Structure):
    pass


def error_check(ret):
    if ret != 0 and ret < len(menoh_error_list):
        get_error = menoh.menoh_get_last_error_message
        get_error.restype = ctypes.c_char_p
        data = get_error()
        raise ValueError("{}\n{}".format(menoh_error_list[ret], data))
    else:
        pass


def main():
    CONV1_1_IN_NAME = b"140326425860192"
    FC6_OUT_NAME = b"140326200777584"
    SOFTMAX_OUT_NAME = b"140326200803680"

    model_data = ctypes.pointer(Handle())
    error_check(menoh.menoh_make_model_data_from_onnx(
        b'data/VGG16.onnx', ctypes.byref(model_data)))

    vpt_builder = ctypes.pointer(Handle())
    error_check(menoh.menoh_make_variable_profile_table_builder(
        ctypes.byref(vpt_builder)))
    error_check(
        menoh.menoh_variable_profile_table_builder_add_input_profile_dims_4(
            vpt_builder, CONV1_1_IN_NAME, 0, 1, 3, 224, 224
        ))
    error_check(
        menoh.menoh_variable_profile_table_builder_add_output_profile(
            vpt_builder, FC6_OUT_NAME, 0
        ))
    error_check(
        menoh.menoh_variable_profile_table_builder_add_output_profile(
            vpt_builder, SOFTMAX_OUT_NAME, 0
        ))

    profile_table = ctypes.pointer(Handle())
    error_check(
        menoh.menoh_build_variable_profile_table(vpt_builder, model_data,
                                                 ctypes.byref(profile_table))
    )

    softmax_out_dim_0 = ctypes.c_int()
    softmax_out_dim_1 = ctypes.c_int()
    error_check(
        menoh.menoh_variable_profile_table_get_dims_at(
            profile_table, SOFTMAX_OUT_NAME, 0, ctypes.byref(softmax_out_dim_0)
        ))
    error_check(
        menoh.menoh_variable_profile_table_get_dims_at(
            profile_table, SOFTMAX_OUT_NAME, 1, ctypes.byref(softmax_out_dim_1)
        ))
    error_check(
        menoh.menoh_model_data_optimize(model_data, profile_table))

    model_builder = ctypes.pointer(Handle())
    error_check(
        menoh.menoh_make_model_builder(profile_table,
                                       ctypes.byref(model_builder)))

    input_buff = ctypes.pointer((ctypes.c_float * (1 * 3 * 224 * 224))())
    error_check(menoh.menoh_model_builder_attach_external_buffer(
        model_builder, CONV1_1_IN_NAME, input_buff))

    model = ctypes.pointer(Handle())
    error_check(menoh.menoh_build_model(model_builder, model_data, b"mkldnn",
                                        b"", ctypes.byref(model)))
    error_check(menoh.menoh_delete_model_data(model_data))

    fc6_output_buff = ctypes.pointer((ctypes.c_float * 10)())
    error_check(menoh.menoh_model_get_variable_buffer_handle(
        model, FC6_OUT_NAME, ctypes.byref(fc6_output_buff)))

    softmax_output_buff = ctypes.pointer((
        ctypes.c_float * (softmax_out_dim_0.value *
                          softmax_out_dim_1.value))())
    error_check(menoh.menoh_model_get_variable_buffer_handle(
        model, SOFTMAX_OUT_NAME, ctypes.byref(softmax_output_buff)))

    for i in range(1 * 3 * 224 * 224):
        input_buff.contents[i] = 0.5

    error_check(menoh.menoh_model_run(model))

    for i in range(10):
        print(fc6_output_buff.contents[i], end=" ")
    print("")

    for i in range(softmax_out_dim_0.value):
        for j in range(softmax_out_dim_1.value):
            print(softmax_output_buff.contents[i * softmax_out_dim_1.value + j]
                  , end=" ")
        print("")

    error_check(menoh.menoh_delete_model(model))
    error_check(menoh.menoh_delete_model_builder(model_builder))
    error_check(menoh.menoh_delete_variable_profile_table_builder(vpt_builder))


if __name__ == '__main__':
    main()
