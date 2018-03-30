// ------------------------------------------------------------------
// Project: Mask R-CNN
// File: ROIAlignLayer
// Adopted from roi_pooling_layer.cpp (written by Ross Grischik)
// ------------------------------------------------------------------

#include <cfloat>
#include <algorithm>
#include <stdlib.h>
#include <string>
#include <utility>
#include <vector>

//#include "caffe/blob.hpp"
//#include "caffe/common.hpp"
//#include "caffe/layer.hpp"
//#include "caffe/proto/caffe.pb.h"
#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;
using std::vector;
using std::fabs;
namespace caffe {

template <typename Dtype>
Dtype Bilinear_interpolate_cpu(const Dtype* bottom_data, const int height, const int width, Dtype y, Dtype x) { 
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
void Bilinear_interpolate_gradient_cpu(const int height, const int width, Dtype y, Dtype x,
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
void ROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top)
{
  ROIAlignParameter roi_align_param = this->layer_param_.roi_align_param();
  CHECK_GT(roi_align_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_align_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_align_param.pooled_h();
  pooled_width_ = roi_align_param.pooled_w();
  spatial_scale_ = roi_align_param.spatial_scale();
  sampling_ratio_ = 2;//roi_align_param.sampling_ratio(); //-----------------------------
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top)
{
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  int shape_init[] = {bottom[1]->num(), channels_, pooled_height_,
      pooled_width_, 4};
  const vector<int> shape(shape_init, shape_init + sizeof(shape_init)
      / sizeof(int));
  //max_mult_.Reshape(shape);
  //max_pts_.Reshape(shape);
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top)
{
  LOG(INFO) << "DOING CPU FORWARD NOW ";
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  CHECK_EQ(bottom[1]->offset(1),5);
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  //std::cout << "TOTAL = " << num_rois*channels_*height_*width_ << "\n";
  // For each ROI R = [batch_index x1 y1 x2 y2]:
  for (int n = 0; n < num_rois; ++n) {

    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = bottom_rois[1] * spatial_scale_;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale_;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale_;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale_;
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    Dtype roi_height = max(roi_end_h - roi_start_h, Dtype(1.0));
    Dtype roi_width = max(roi_end_w - roi_start_w, Dtype(1.0));
    const Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)  /  static_cast<Dtype>(pooled_width_);

    int sampling_w = (sampling_ratio_ > 0) ? sampling_ratio_ : ceil(roi_height / pooled_height_);
    int sampling_h = (sampling_ratio_ > 0) ? sampling_ratio_ : ceil(roi_width / pooled_width_);
    const Dtype count = sampling_w * sampling_h;

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          Dtype output = Dtype(0.0);
          for (int i_y = 0; i_y < sampling_h; i_y++){
            const Dtype y = roi_start_h + ph * bin_size_h + 
                static_cast<Dtype>(i_y + Dtype(0.5)) * bin_size_h / static_cast<Dtype>(sampling_h);
            for (int i_x = 0; i_x < sampling_w; i_x++){
              const Dtype x = roi_start_w + pw * bin_size_w + 
                  static_cast<Dtype>(i_x + Dtype(0.5)) * bin_size_w / static_cast<Dtype>(sampling_w);
              Dtype value = Bilinear_interpolate_cpu(batch_data, height_, width_, y, x); 
              output += value;
            } //i_x
          } //i_y
          output /= count;
          top_data[ph * pooled_width_ + pw] = output;
        } //pw
      } // ph
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    } // channels
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }//num_rois
  Dtype asum=caffe_cpu_asum(top[0]->count(),top[0]->cpu_data());
  LOG(INFO)<<asum;
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* bottom_rois = bottom[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int count = bottom[0]->count();
  caffe_set(count, Dtype(0.), bottom_diff);

  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();

  for (int n = 0; n < num_rois; ++n) {

    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = bottom_rois[1] * spatial_scale_;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale_;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale_;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale_;
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    Dtype roi_height = max(roi_end_h - roi_start_h, Dtype(1.0));
    Dtype roi_width = max(roi_end_w - roi_start_w, Dtype(1.0));
    const Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)  /  static_cast<Dtype>(pooled_width_);

    int sampling_w = (sampling_ratio_ > 0) ? sampling_ratio_: ceil(roi_height / pooled_height_);
    int sampling_h = (sampling_ratio_ > 0) ? sampling_ratio_ : ceil(roi_width / pooled_width_);
    const Dtype count = sampling_w * sampling_h;

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          const Dtype top_diff_this_bin = top_diff[ph * pooled_width_ + pw];
          Dtype w1, w2, w3, w4;
          int x_low, x_high, y_low, y_high;
          for (int i_y = 0; i_y < sampling_h; i_y++){
            const Dtype y = roi_start_h + ph * bin_size_h + 
                static_cast<Dtype>(i_y + Dtype(0.5)) * bin_size_h / static_cast<Dtype>(sampling_h);
            for (int i_x = 0; i_x < sampling_w; i_x++){
              const Dtype x = roi_start_w + pw * bin_size_w + 
                  static_cast<Dtype>(i_x + Dtype(0.5)) * bin_size_w / static_cast<Dtype>(sampling_w);

              Bilinear_interpolate_gradient_cpu(height_, width_, y, x,
                        w1, w2, w3, w4, x_low, x_high, y_low, y_high); 

              Dtype g1 = top_diff_this_bin * w1 / count;
              Dtype g2 = top_diff_this_bin * w2 / count;
              Dtype g3 = top_diff_this_bin * w3 / count;
              Dtype g4 = top_diff_this_bin * w4 / count;

              if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
                bottom_diff[y_low * width_ + x_low] += g1;
                bottom_diff[y_low * width_ + x_high] += g2;
                bottom_diff[y_high * width_ + x_low] += g3;
                bottom_diff[y_high * width_ + x_high] += g4;
              } //if
            } //i_x
          } //i_y 
        } //pw
      } // ph
      // Increment all data pointers by one channel
      bottom_diff += bottom[0]->offset(0, 1);
      top_diff += top[0]->offset(0, 1);
    } // channels
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }//num_rois
}


#ifdef CPU_ONLY
STUB_GPU(ROIAlignLayer);
#endif

INSTANTIATE_CLASS(ROIAlignLayer);
REGISTER_LAYER_CLASS(ROIAlign);

}  // namespace caffe
