//==============================================================
// DPC++ Example
//
// Image Convoluton with DPC++
//
// Author: Yan Luo
//
// Copyright ©  2020-
//
// MIT License
//
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include "dpc_common.hpp"
#if FPGA || FPGA_EMULATOR || FPGA_PROFILE
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;

// useful header files for image convolution
#include "utils.h"
#include "bmp-utils.h"

/*for stats aftewards*/
using Duration = std::chrono::duration<double>;
class Timer {
 public:
  Timer() : start(std::chrono::steady_clock::now()) {}

  Duration elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start);
  }

 private:
  std::chrono::steady_clock::time_point start;
};

static const char* inputImagePath = "./Images/cat.bmp";

//************************************
// Image Rotation in DPC++ on device: 
//************************************
void ImageRotate_v1(queue &q, float *image_in, float *image_out, float sinTheta, 
    float cosTheta, const size_t ImageRows, const size_t ImageCols) 
{

  // We create buffers for the input and output data.
  //
  buffer<float, 1> image_in_buf(image_in, range<1>(ImageRows*ImageCols));
  buffer<float, 1> image_out_buf(image_out, range<1>(ImageRows*ImageCols));

  // Create the range object for the pixel data.
  range<2> num_items{ImageRows, ImageCols};

  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  q.submit([&](handler &h) {
    // Create an accessor to buffers with access permission: read, write or
    // read/write. The accessor is a way to access the memory in the buffer.
    accessor srcPtr(image_in_buf, h, read_only);

    // Another way to get access is to call get_access() member function 
    auto dstPtr = image_out_buf.get_access<access::mode::write>(h);

    // Use parallel_for to run image convolution in parallel on device. This
    // executes the kernel.
    //    1st parameter is the number of work items. **stay the same**
    //    2nd parameter is the kernel, a lambda that specifies what to do per
    //    work item. The parameter of the lambda is the work item id.
    // DPC++ supports unnamed lambda kernel by default.
    //"item" is an instance of the kernel with a 2D id
    //num_items is the "range"
    h.parallel_for(num_items, [=](id<2> item) 
    { 

      // get row and col of the pixel assigned to this work item
      //this is 2D, but output image buffer is 1D so it has
      //to be translated as row*ImageCols+col to pull from input buffer later
      //grab the (x,y) position of the current pixel: (col, row)
      int row = item[1];  //y position
      int col = item[0];  //x position
      //graphical adjustment
      //int row_adj = ImageRows-row;
      /* calculate location of data to move int (row, col)
      * output decomposition as mentioned*/
      float xpos = ((float)col)*cosTheta + ((float)row)*sinTheta;
      float ypos = -1.0f*((float)col)*sinTheta + ((float)row)*cosTheta;
      //bmp adjustment
      //float ypos_adj = ImageRows - ypos;

        /* Bound checking: dont write the new pixel if its out of image bounds, just cut off*/
       if(((int)xpos >= 0) && ((int)xpos < ImageCols) &&
          ((int)ypos >= 0) && ((int)ypos < ImageRows) )
       {
          /* read (row,col) src data and store at (xpos,ypos)
           * in dest data
           * in this case, because we rotate about the origin and
           * there is no translation, we know that (xpos, ypos) will be
           * unique for each input (row, col) and so each work-item can
           * write its results independently*/
           dstPtr[(int)ypos * ImageCols + (int)xpos] = srcPtr[row*ImageCols+col];
       }
    }); //end of parallel_for
  });   //end of q.submit
}


int main() {
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA || FPGA_PROFILE
  // DPC++ extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  default_selector d_selector;
#endif

  float *hInputImage;
  float *hOutputImage;

  /*rows and columns values come from readBmp function later*/
  int imageRows;
  int imageCols;
  int i;

  /* Theta = 315 degrees same as openCL example*/
  //float sinTheta = -0.70710678118;
  //float cosTheta = 0.70710678118;
  /*try 90 degrees*/
  float sinTheta = -0.70710678118;;
  float cosTheta = 0.70710678118;;

#ifndef FPGA_PROFILE
  // Query about the platform
  unsigned number = 0;
  auto myPlatforms = platform::get_platforms();
  // loop through the platforms to poke into
  for (auto &onePlatform : myPlatforms) {
    std::cout << ++number << " found .." << std::endl << "Platform: " 
    << onePlatform.get_info<info::platform::name>() <<std::endl;
    // loop through the devices
    auto myDevices = onePlatform.get_devices();
    for (auto &oneDevice : myDevices) {
      std::cout << "Device: " 
      << oneDevice.get_info<info::device::name>() <<std::endl;
    }
  }
  std::cout<<std::endl;
#endif


  /* Read in the BMP image */
  hInputImage = readBmpFloat(inputImagePath, &imageRows, &imageCols);
  printf("imageRows=%d, imageCols=%d\n", imageRows, imageCols);
  /* Allocate space for the output image */
  hOutputImage = (float *)malloc( imageRows*imageCols * sizeof(float) );
  /*****what is this for loop about****initialization but why 1234.0? dummy?*/
  for(i=0; i<imageRows*imageCols; i++)
    hOutputImage[i] = 1234.0;


  Timer t;

  try {
    /*initialization of a queue called q for default device with exceptions*/
    queue q(d_selector, dpc_common::exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

    // Image Rotation in DPC++
    ImageRotate_v1(q, hInputImage, hOutputImage, sinTheta, cosTheta, imageRows, imageCols);
  } catch (exception const &e) {
    std::cout << "An exception is caught for image rotation.\n";
    std::terminate();
  }

  std::cout << t.elapsed().count() << " seconds\n";

  /* Save the output bmp */
  printf("Output image saved as: cat-rotated.bmp\n");
  writeBmpFloat(hOutputImage, "cat-rotated.bmp", imageRows, imageCols,
          inputImagePath);

/*i removed verification bc there is no image rotation verification in gold.h*/

  return 0;
}