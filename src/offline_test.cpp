/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
////////////////////
// This is modified from Autoware point_pillars_ros.cpp file
///////////////////
// headers in STL
#include <chrono>
#include <cmath>

#include <gtest/gtest.h>

// headers in PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>

#include "lidar_point_pillars/point_pillars.h"

#include <string>
#include <memory>
#include <iterator>
#include <algorithm>

const int NUM_POINT_FEATURE_ = 4;
const float NORMALIZING_INTENSITY_VALUE_ = 255.0f;
const float offset_z = 0.0f;

template<class T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    std::copy(v.cbegin(), v.cend(),
              std::ostream_iterator<T>(out, " "));
    return out;
}

void pclToArray(const pcl::PointCloud<pcl::PointXYZI>::Ptr& in_pcl_pc_ptr, float* out_points_array,
                                 const float offset_z)
{
  for (size_t i = 0; i < in_pcl_pc_ptr->size(); i++)
  {
    pcl::PointXYZI point = in_pcl_pc_ptr->at(i);
    out_points_array[i * NUM_POINT_FEATURE_ + 0] = point.x;
    out_points_array[i * NUM_POINT_FEATURE_ + 1] = point.y;
    out_points_array[i * NUM_POINT_FEATURE_ + 2] = point.z + offset_z;
    out_points_array[i * NUM_POINT_FEATURE_ + 3] = float(point.intensity / NORMALIZING_INTENSITY_VALUE_);
  }
}

int main() {
    std::unique_ptr<PointPillars> point_pillars_ptr_;
    std::unique_ptr<PreprocessPoints> preprocess_points_ptr_;

    bool reproduce_result_mode=false;
    float score_threshold = 0.5;
    float nms_overlap_threshold = 0.5;
    std::string pfe_onnx_file = 
            "/mnt/raid1/Research/Autoware_inference/kitti_pretrained_point_pillars/pfe.onnx";
    std::string rpn_onnx_file = 
            "/mnt/raid1/Research/Autoware_inference/kitti_pretrained_point_pillars/rpn.onnx";



    point_pillars_ptr_.reset(new PointPillars(reproduce_result_mode, 
                                              score_threshold, 
                                              nms_overlap_threshold,
                                              pfe_onnx_file, 
                                              rpn_onnx_file));

    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_ptr(new pcl::PointCloud<pcl::PointXYZI>);

    // Read test file 000003.pcd
    if (pcl::io::loadPCDFile<pcl::PointXYZI>("/home/saiclei/Research/Codes/kitti-pcl/src/build/000003.pcd", 
                                               *pcl_pc_ptr) == -1) {
        PCL_ERROR("Couldn't read file 000003.pcd. \n");
        return -1;
    }

    std::cout << "Loaded " << pcl_pc_ptr->width * pcl_pc_ptr->height
                           << " data points\n ";

    std::cout << "The size is: " << pcl_pc_ptr->size() << '\n';
    float* points_array = new float[pcl_pc_ptr->size() * NUM_POINT_FEATURE_];

    pclToArray(pcl_pc_ptr, points_array, offset_z);
    std::vector<float> out_detection;

    auto start = std::chrono::high_resolution_clock::now();
    point_pillars_ptr_->doInference(points_array, pcl_pc_ptr->size(), out_detection);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time is: " << elapsed.count() << " s\n";
  
    std::cout << "The sizez of out_detection is: " << out_detection.size() << std::endl; 
    std::cout << out_detection << std::endl; 
    delete[] points_array;

}
