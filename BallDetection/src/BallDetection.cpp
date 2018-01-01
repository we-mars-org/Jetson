#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

//
///** @function main */
//int main(int argc, char** argv)
//{
//	cv::Mat src, src_gray;
//
//	/// Read the image
//	//src = cv::imread(argv[1], 1);
//	src = cv::imread("C:/Users/djorna.Pokedex/Documents/ZED/BallDetection/build/used-tennis-balls.jpg");
//
//	if (!src.data)
//	{
//		return -1;
//	}
//
//	/// Convert it to gray
//	cv::cvtColor(src, src_gray, CV_BGR2GRAY);
//
//	/// Reduce the noise so we avoid false circle detection
//	cv::GaussianBlur(src_gray, src_gray, cv::Size(9, 9), 2, 2);
//
//	std::vector<cv::Vec3f> circles;
//
//	/// Apply the Hough Transform to find the circles
//	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 8, 20, 10, 0, 0);
//
//	/// Draw the circles detected
//	for (size_t i = 0; i < circles.size(); i++)
//	{
//		cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
//		int radius = cvRound(circles[i][2]);
//		// circle center
//		circle(src, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
//		// circle outline
//		circle(src, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
//	}
//
//	/// Show your results
//	cv::namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
//	imshow("Hough Circle Transform Demo", src);
//
//	cv::waitKey(0);
//	return 0;
//}

///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2017, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/***********************************************************************************************
** This sample demonstrates how to use the ZED SDK with OpenCV. 					  	      **
** Depth and images are captured with the ZED SDK, converted to OpenCV format and displayed. **
***********************************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>

// Sample includes
#include <SaveDepth.hpp>

using namespace sl;

cv::Mat slMat2cvMat(Mat& input);
void printHelp();
cv::Point findTennisBall(cv::Mat im);


int main(int argc, char **argv) {

	// Create a ZED camera object
	Camera zed;

	// Set configuration parameters
	InitParameters init_params;
	init_params.camera_resolution = RESOLUTION_VGA;
	init_params.depth_mode = DEPTH_MODE_PERFORMANCE;
	init_params.coordinate_units = UNIT_METER;
	init_params.camera_fps = 100;

	// Open the camera
	ERROR_CODE err = zed.open(init_params);
	if (err != SUCCESS) {
		printf("%s\n", errorCode2str(err).c_str());
		zed.close();
		return 1; // Quit if an error occurred
	}

	// Display help in console
	printHelp();

	// Set runtime parameters after opening the camera
	RuntimeParameters runtime_parameters;
	runtime_parameters.sensing_mode = SENSING_MODE_STANDARD;

	// Prepare new image size to retrieve half-resolution images
	Resolution image_size = zed.getResolution();
	int new_width = image_size.width / 2;
	int new_height = image_size.height / 2;

	// To share data between sl::Mat and cv::Mat, use slMat2cvMat()
	// Only the headers and pointer to the sl::Mat are copied, not the data itself
	Mat image_zed(new_width, new_height, MAT_TYPE_8U_C4);
	cv::Mat image_ocv = slMat2cvMat(image_zed);
	Mat depth_image_zed(new_width, new_height, MAT_TYPE_8U_C4);
	cv::Mat depth_image_ocv = slMat2cvMat(depth_image_zed);
	sl::Mat normal_image_zed(new_width, new_height, sl::MAT_TYPE_8U_C4);
	cv::Mat normal_image_ocv = slMat2cvMat(normal_image_zed);
	
	//ball detection vars
	int hmin = 30, smin = 68, vmin = 115, hmax = 62, smax = 231, vmax = 255;
	cv::Mat mask = image_ocv;
	std::vector<cv::Vec3f> circles;	

	//Create trackbars
	cv::namedWindow("Adjust");
	cv::createTrackbar("H-", "Adjust", &hmin, 360);
	cv::createTrackbar("S-", "Adjust", &smin, 255);
	cv::createTrackbar("V-", "Adjust", &vmin, 255);
	cv::createTrackbar("H+", "Adjust", &hmax, 360);
	cv::createTrackbar("S+", "Adjust", &smax, 255);
	cv::createTrackbar("V+", "Adjust", &vmax, 255);

	std::vector<cv::Vec4i> hierarchy;
	std::vector<std::vector<cv::Point> > contours;
	cv::Mat canny_output;
	sl::Mat depth,point_cloud;

	// Loop until 'q' is pressed
	char key = ' ';
	while (key != 'q') {

		if (zed.grab(runtime_parameters) == SUCCESS) {

			// Retrieve the left image, depth image in half-resolution
			zed.retrieveImage(image_zed, VIEW_LEFT, MEM_CPU, new_width, new_height);
			zed.retrieveImage(depth_image_zed, VIEW_DEPTH, MEM_CPU, new_width, new_height);
			//zed.retrieveImage(normal_image_zed, VIEW_NORMALS);

			// Retrieve the RGBA point cloud in half-resolution
			// To learn how to manipulate and display point clouds, see Depth Sensing sample
			zed.retrieveMeasure(depth, MEASURE_DEPTH,MEM_CPU,new_width,new_height);
			zed.retrieveMeasure(point_cloud, MEASURE_XYZRGBA,MEM_CPU, new_width, new_height);

			//filter out yellow
			cv::cvtColor(image_ocv,mask,cv::COLOR_BGR2HSV);
			//cv::medianBlur(mask,mask, 11);
			cv::inRange(mask,cv::Scalar(hmin,smin,vmin),cv::Scalar(hmax, smax, vmax), mask);
			//cv::GaussianBlur(mask, mask, cv::Size(3, 3), 0);
			cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)),cv::Point(-1,-1),5);
			//cv::medianBlur(mask, mask, 5);
			cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0);

			cv::Canny(mask, canny_output, 100, 200, 3);
			/// Find contours
			cv::findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

			//iterate over contours
			for (int i = 0; i < contours.size(); i++)
			{
				cv::Point2f center;
				float radius;
				cv::minEnclosingCircle(contours[i], center,radius);

				// Get the moments
				cv::Moments mu = moments(contours[i], false);

				//  Get the mass centers:
				cv::Point2f mc = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);

				//draw enclosing circle
				cv::circle(image_ocv, center, radius, cv::Scalar(0, 255, 255),2);

				//draw centroid
				cv::circle(image_ocv, mc, 3, cv::Scalar(0, 0, 255), 2);

				//get tennis ball location from point cloud
				//sl::float3 depth_value;
				//depth.getValue(center.x, center.y, &depth_value);
				sl::float4 point_cloud_value;				
				point_cloud.getValue(center.x, center.y, &point_cloud_value);
				float distance = sqrt(point_cloud_value.x*point_cloud_value.x + point_cloud_value.y*point_cloud_value.y + point_cloud_value.z*point_cloud_value.z);
				std::string point_cloud_z;
				if (point_cloud_value.z == TOO_FAR)
					point_cloud_z = "Too far";
				else if (point_cloud_value.z == TOO_CLOSE)
					point_cloud_z = "Too close";
				else if (point_cloud_value.z == OCCLUSION_VALUE)
					point_cloud_z = "Occluded";
				else
					point_cloud_z = std::to_string(point_cloud_value.z);

				//write descriptive text
				cv::putText(image_ocv, "x:" + std::to_string(center.x), center+cv::Point2f(0,-20), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));
				cv::putText(image_ocv, "y:" + std::to_string(center.y), center, CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));
				cv::putText(image_ocv, "z:" + point_cloud_z, center+cv::Point2f(0,20), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));
			}

			/// Draw contours
			/*cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);
			for (int i = 0; i< contours.size(); i++)
			{
				cv::drawContours(image_ocv, contours, i,cv::Scalar(0,0,255), 2, 8, hierarchy, 0, cv::Point());
			}*/
			//translate this to c++:
			/*
			cnts = cv2.findContours(yellow_mask.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)[-2]
				center = None

				# only proceed if at least one contour was found
				if len(cnts) > 0:
			# find the largest contour in the mask, then use
				# it to compute the minimum enclosing circle and
				# centroid
				#c = max(cnts, key = cv2.contourArea)
				for c in cnts :
			((x, y), radius) = cv2.minEnclosingCircle(c)
				M = cv2.moments(c)
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

				# only proceed if the radius meets a minimum size
				if radius > 10:
			# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius),
					(0, 255, 255), 2)
				cv2.circle(frame, center, 5, (0, 0, 255), -1)
				*/
			/*
			cv::erode(yellow_mask, yellow_mask, cv::Mat::ones(cv::Size(5, 5), CV_8UC3));
			cv::dilate(yellow_mask, yellow_mask, cv::Mat::ones(cv::Size(5, 5), CV_8UC3));
			cv::bitwise_and(hsv, hsv, yellow_image, yellow_mask);
			cv::HoughCircles(yellow_mask,circles, cv::HOUGH_GRADIENT, 1, 20, 20, 20, 0, 0);

			//draw circles
			for (size_t i = 0; i < circles.size(); i++)
			{
				cv::Vec3i c = circles[i];
				cv::circle(image_ocv, cv::Point(c[0], c[1]), c[2], cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
				cv::circle(image_ocv, cv::Point(c[0], c[1]), 2, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
			}
			*/

			// Display image and depth using cv:Mat which share sl:Mat data
			cv::imshow("Image", image_ocv);
			cv::imshow("Depth", depth_image_ocv);
			//cv::imshow("Normals", normal_image_ocv);
			cv::imshow("Mask", mask);

			// Handle key event
			key = cv::waitKey(10);
			processKeyEvent(zed, key);
		}
	}
	zed.close();
	return 0;
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(Mat& input) {
	// Mapping between MAT_TYPE and CV_TYPE
	int cv_type = -1;
	switch (input.getDataType()) {
	case MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
	case MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
	case MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
	case MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
	case MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
	case MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
	case MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
	case MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
	default: break;
	}

	// Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
	// cv::Mat and sl::Mat will share a single memory structure
	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}

/**
* This function displays help in console
**/
void printHelp() {
	std::cout << " Press 's' to save Side by side images" << std::endl;
	std::cout << " Press 'p' to save Point Cloud" << std::endl;
	std::cout << " Press 'd' to save Depth image" << std::endl;
	std::cout << " Press 'm' to switch Point Cloud format" << std::endl;
	std::cout << " Press 'n' to switch Depth format" << std::endl;
}

cv::Point findTennisBall(cv::Mat im) {
	return cv::Point(0,0);
}