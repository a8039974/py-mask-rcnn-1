// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/fast_rcnn_layers.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__device__ Dtype bilinear_interpolate(const Dtype* bottom_data, const int height, const int width, Dtype y, Dtype x) { 
  if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;
  if (y <= 0) y = 0; if (x <= 0) x = 0;
  
  int y_low = (int)y;      //     x_low       x_high
  int x_low = (int)x;      // y_low
  int y_high;              //           (y,x)
  int x_high;              // y_high

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (Dtype)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (Dtype)x_low;
  } else {
    x_high = x_low + 1;
  }

  Dtype ly = y - y_low, lx = x - x_low;
  Dtype hy = Dtype(1.0) - ly, hx = Dtype(1.0) - lx;

  Dtype v1 = bottom_data[y_low * width + x_low];
  Dtype v2 = bottom_data[y_low * width + x_high];
  Dtype v3 = bottom_data[y_high * width + x_low];
  Dtype v4 = bottom_data[y_high * width + x_high];
  Dtype w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  Dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename Dtype>
__device__ void bilinear_interpolate_gradient(const int height, const int width, Dtype y, Dtype x,
      Dtype& w1, Dtype& w2, Dtype& w3, Dtype& w4, int& x_low, int& x_high, int& y_low, int& y_high) { 
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;
  
  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (Dtype)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (Dtype)x_low;
  } else {
    x_high = x_low + 1;
  }

  Dtype ly = y - y_low, lx = x - x_low;
  Dtype hy = Dtype(1.0) - ly, hx = Dtype(1.0) - lx;

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename Dtype>
__global__ void ROIAlignForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int sampling_ratio_, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w, Dtype(1.0));
    Dtype roi_height = max(roi_end_h - roi_start_h, Dtype(1.0));
    Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

    bottom_data += (roi_batch_ind * channels + c) * height * width;

    int sampling_w = (sampling_ratio_ > 0) ? sampling_ratio_ : ceil(roi_height / pooled_height);
    int sampling_h = (sampling_ratio_ > 0) ? sampling_ratio_ : ceil(roi_width / pooled_width);
    const Dtype count = sampling_w * sampling_h;

    Dtype output = Dtype(0.0);
    for (int i_y = 0; i_y < sampling_h; i_y++){
      const Dtype y = roi_start_h + ph * bin_size_h + 
          static_cast<Dtype>(i_y + Dtype(0.5)) * bin_size_h / static_cast<Dtype>(sampling_h);
      for (int i_x = 0; i_x < sampling_w; i_x++){
        const Dtype x = roi_start_w + pw * bin_size_w + 
            static_cast<Dtype>(i_x + Dtype(0.5)) * bin_size_w / static_cast<Dtype>(sampling_w);

        Dtype value = bilinear_interpolate(bottom_data, height, width, y, x);
        output += value;
      } //i_x
    } //i_y
    output /= count;
    top_data[index] = output;
  } //CUDA_KERNEL_LOOP
} //ROIAlignForward

template <typename Dtype>
__global__ void ROIAlignBackward(const int nthreads, const Dtype* top_diff,
    const Dtype spatial_scale, const int sampling_ratio_, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff, const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) coords the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w, Dtype(1.0));
    Dtype roi_height = max(roi_end_h - roi_start_h, Dtype(1.0));
    Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

    bottom_diff += (roi_batch_ind * channels + c) * height * width;
    top_diff += (n * channels + c) * pooled_height * pooled_width;
    const Dtype top_diff_this_bin = top_diff[ph * pooled_width + pw];

    int sampling_w = (sampling_ratio_ > 0) ? sampling_ratio_ : ceil(roi_height / pooled_height);
    int sampling_h = (sampling_ratio_ > 0) ? sampling_ratio_ : ceil(roi_width / pooled_width);
    const Dtype count = sampling_w * sampling_h;

    Dtype w1, w2, w3, w4;
    int x_low, x_high, y_low, y_high;
    for (int i_y = 0; i_y < sampling_h; i_y++){
      const Dtype y = roi_start_h + ph * bin_size_h + 
          static_cast<Dtype>(i_y + Dtype(0.5)) * bin_size_h / static_cast<Dtype>(sampling_h);
      for (int i_x = 0; i_x < sampling_w; i_x++){
        const Dtype x = roi_start_w + pw * bin_size_w + 
            static_cast<Dtype>(i_x + Dtype(0.5)) * bin_size_w / static_cast<Dtype>(sampling_w);

        bilinear_interpolate_gradient(height, width, y, x,
                  w1, w2, w3, w4, x_low, x_high, y_low, y_high);

        Dtype g1 = top_diff_this_bin * w1 / count;
        Dtype g2 = top_diff_this_bin * w2 / count;
        Dtype g3 = top_diff_this_bin * w3 / count;
        Dtype g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          caffe_gpu_atomic_add(g1, bottom_diff + y_low * width + x_low);
          caffe_gpu_atomic_add(g2, bottom_diff + y_low * width + x_high);
          caffe_gpu_atomic_add(g3, bottom_diff + y_high * width + x_low);
          caffe_gpu_atomic_add(g4, bottom_diff + y_high * width + x_high);
        } //if
      } //i_x
    } //i_y 
  } //CUDA_KERNEL_LOOP
} //ROIAlignBackward


template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();

  ROIAlignForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, sampling_ratio_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // if (propagate_down[1]) {
  //   LOG(FATAL) << this->type()
  //     << " Layer cannot backpropagate to label inputs.";
  // }
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = top[0]->count();
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, spatial_scale_, sampling_ratio_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignLayer);

}  // namespace caffe
