// YOLO OpenCL Preprocessing Kernel

__kernel void yolo_preprocess_kernel(
    __global const uchar *input,
    __global float *output,
    const int in_width,
    const int in_height,
    const int in_stride,
    const int out_width,
    const int out_height,
    const int pad_x,
    const int pad_y,
    const float scale_x,
    const float scale_y,
    const float norm_scale
) {
    int x = get_global_id(0); // target width index
    int y = get_global_id(1); // target height index

    if (x >= out_width || y >= out_height) {
        return;
    }

    // Output is CHW
    int out_c0_idx = 0 * (out_width * out_height) + y * out_width + x;
    int out_c1_idx = 1 * (out_width * out_height) + y * out_width + x;
    int out_c2_idx = 2 * (out_width * out_height) + y * out_width + x;

    // Check padding bounds (Letterbox)
    if (x < pad_x || x >= (out_width - pad_x) || y < pad_y || y >= (out_height - pad_y)) {
        // Pad with (114, 114, 114) normalized
        float pad_val = 114.0f * norm_scale;
        output[out_c0_idx] = pad_val;
        output[out_c1_idx] = pad_val;
        output[out_c2_idx] = pad_val;
        return;
    }

    // Map output (x,y) back to input (src_x, src_y)
    // Note: scale_x and scale_y are (in / out_active)
    float src_x_f = (x - pad_x + 0.5f) * scale_x - 0.5f;
    float src_y_f = (y - pad_y + 0.5f) * scale_y - 0.5f;

    int src_x = (int)round(src_x_f);
    int src_y = (int)round(src_y_f);

    // Clamp
    src_x = max(0, min(src_x, in_width - 1));
    src_y = max(0, min(src_y, in_height - 1));

    // Input is BGR, HWC
    int in_idx = src_y * in_stride + src_x * 3;
    uchar b = input[in_idx + 0];
    uchar g = input[in_idx + 1];
    uchar r = input[in_idx + 2];

    // BGR -> RGB and normalize
    output[out_c0_idx] = (float)r * norm_scale; // R -> Channel 0
    output[out_c1_idx] = (float)g * norm_scale; // G -> Channel 1
    output[out_c2_idx] = (float)b * norm_scale; // B -> Channel 2
}
