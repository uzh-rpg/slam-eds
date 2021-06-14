/*
* This file is part of the EDS: Event-aided Direct Sparse Odometry
* (https://rpg.ifi.uzh.ch/eds.html)
*
* Copyright (c) 2022 Javier Hidalgo-Carrió, Robotics and Perception
* Group (RPG) University of Zurich.
*
* EDS is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* EDS is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <boost/test/unit_test.hpp>
#include <eds/EDS.h>

#include<iostream>
using namespace eds;
using namespace cv;


std::string colormap_name(int id)
{
    switch(id){
        case COLORMAP_AUTUMN :
            return "COLORMAP_AUTUMN";
        case COLORMAP_BONE :
            return "COLORMAP_BONE";
        case COLORMAP_JET :
            return "COLORMAP_JET";
        case COLORMAP_WINTER :
            return "COLORMAP_WINTER";
        case COLORMAP_RAINBOW :
            return "COLORMAP_RAINBOW";
        case COLORMAP_OCEAN :
            return "COLORMAP_OCEAN";
        case COLORMAP_SUMMER:
            return "COLORMAP_SUMMER";
        case COLORMAP_SPRING :
            return "COLORMAP_SPRING";
        case COLORMAP_COOL :
            return "COLORMAP_COOL";
        case COLORMAP_HSV :
            return "COLORMAP_HSV";
        case COLORMAP_PINK :
            return "COLORMAP_PINK";
        case COLORMAP_HOT :
            return "COLORMAP_HOT";
    }

    return "NONE";
}


BOOST_AUTO_TEST_CASE(test_color_map)
{
    // Read 8-bit grayscale image
    Mat im = imread("test/data/pluto.jpg", IMREAD_GRAYSCALE);
    BOOST_TEST_MESSAGE("img size"<<im.size());

    Mat im_out = Mat::zeros(600, 800, CV_8UC3);

    for (int i=0; i < 4; i++)
    {
        for(int j=0; j < 3; j++)
        {
            int k = i + j * 4;
            Mat im_color = im_out(Rect(i * 200, j * 200, 200, 200));
            applyColorMap(im, im_color, k);
            putText(im_color, colormap_name(k), Point(30, 180), cv::FONT_HERSHEY_DUPLEX, 0.5, Scalar::all(255), 1, cv::LINE_AA);
        }
    }

    BOOST_CHECK (im_out.total() > 0);
    cv::imwrite("test/data/out_colormaps.jpg", im_out);
}

Mat applyCustomColorMap(Mat& im_gray)
{
    unsigned char g[] = {255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,253,251,249,247,245,242,241,238,237,235,233,231,229,227,225,223,221,219,217,215,213,211,209,207,205,203,201,199,197,195,193,191,189,187,185,183,181,179,177,175,173,171,169,167,165,163,161,159,157,155,153,151,149,147,145,143,141,138,136,134,132,131,129,126,125,122,121,118,116,115,113,111,109,107,105,102,100,98,97,94,93,91,89,87,84,83,81,79,77,75,73,70,68,66,64,63,61,59,57,54,52,51,49,47,44,42,40,39,37,34,33,31,29,27,25,22,20,18,17,14,13,11,9,6,4,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


    unsigned char b[] = {255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243,
       242, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230,
       229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217,
       216, 215, 214, 213, 212, 211, 210, 209, 208, 207, 206, 205, 204,
       203, 202, 201, 200, 199, 198, 197, 196, 195, 194, 193, 192, 191,
       190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178,
       177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165,
       164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152,
       151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139,
       138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126,
       125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113,
       112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100,
        99,  98,  97,  96,  95,  94,  93,  92,  91,  90,  89,  88,  87,
        86,  85,  84,  83,  82,  81,  80,  79,  78,  77,  76,  75,  74,
        73,  72,  71,  70,  69,  68,  67,  66,  65,  64,  63,  62,  61,
        60,  59,  58,  57,  56,  55,  54,  53,  52,  51,  50,  49,  48,
        47,  46,  45,  44,  43,  42,  41,  40,  39,  38,  37,  36,  35,
        34,  33,  32,  31,  30,  29,  28,  27,  26,  25,  24,  23,  22,
        21,  20,  19,  18,  17,  16,  15,  14,  13,  12,  11,  10,   9,
         8,   7,   6,   5,   4,   3,   2,   1, 0};

     unsigned char r[] = {255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243,
       242, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230,
       229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217,
       216, 215, 214, 213, 212, 211, 210, 209, 208, 207, 206, 205, 204,
       203, 202, 201, 200, 199, 198, 197, 196, 195, 194, 193, 192, 191,
       190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178,
       177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165,
       164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152,
       151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139,
       138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126,
       125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113,
       112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100,
        99,  98,  97,  96,  95,  94,  93,  92,  91,  90,  89,  88,  87,
        86,  85,  84,  83,  82,  81,  80,  79,  78,  77,  76,  75,  74,
        73,  72,  71,  70,  69,  68,  67,  66,  65,  64,  63,  62,  61,
        60,  59,  58,  57,  56,  55,  54,  53,  52,  51,  50,  49,  48,
        47,  46,  45,  44,  43,  42,  41,  40,  39,  38,  37,  36,  35,
        34,  33,  32,  31,  30,  29,  28,  27,  26,  25,  24,  23,  22,
        21,  20,  19,  18,  17,  16,  15,  14,  13,  12,  11,  10,   9,
         8,   7,   6,   5,   4,   3,   2,   1, 0};

    
    Mat channels[] = {Mat(256,1, CV_8U, b), Mat(256,1, CV_8U, g), Mat(256,1, CV_8U, r)};
    Mat lut; // Create a lookup table
    merge(channels, 3, lut);
    
    Mat im_color;
    LUT(im_gray, lut, im_color);
    
    return im_color;

}

BOOST_AUTO_TEST_CASE(test_color_map_custom)
{
    // Read 8-bit grayscale image
    Mat im = imread("test/data/pluto.jpg", IMREAD_GRAYSCALE);
    cvtColor(im.clone(), im, COLOR_GRAY2BGR);
    Mat im_color = applyCustomColorMap(im);
    
    BOOST_CHECK (im_color.total() > 0);
    cv::imwrite("test/data/out_custom_colormap.jpg", im_color);
}

BOOST_AUTO_TEST_CASE(test_color_map_utils)
{
    // Read 8-bit grayscale image
    Mat im = imread("test/data/pluto.jpg", IMREAD_GRAYSCALE);
    Mat im_32;

    im.convertTo(im_32, CV_32FC1);

    Mat color; eds::utils::color_map(im_32, color, COLORMAP_JET);
    BOOST_CHECK (color.total() > 0);
    cv::imwrite("test/data/out_color_map.jpg", color);
}

BOOST_AUTO_TEST_CASE(test_color_lut)
{
    int divideWidth = 30;
	uchar table1[256];
	uchar table3[256*3];
	for (int i = 0; i < 256; ++i) {
		table1[i] = divideWidth * (i / divideWidth);
	}
	for (int i = 0; i < 256; ++i) {
		table3[i * 3] = divideWidth * (i / divideWidth);
		table3[i * 3 + 1] = divideWidth * (i / divideWidth);
		table3[i * 3 + 2] = divideWidth * (i / divideWidth);
	}
 
	// Create a lookupTable of type Mat
	cv::Mat lookupTable1(1, 256, CV_8U, table1);
	cv::Mat lookupTable3(1, 256, CV_8UC3, table3);

    cv::Vec3b value(0, 100, 100);
    cv::Vec3b color_value;
    cv::LUT(value, lookupTable1, color_value);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] value "<< value<<" -> "<<color_value);


    ::eds::utils::BlueWhiteRed cm;
    cm.init(256);
    //(cm)(value, color_value); // this does not work
    //BOOST_TEST_MESSAGE("[BOOST MESSAGE] value "<< value<<" -> "<<color_value);
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] value "<< 0.0<<" -> "<<::eds::utils::valueToColor(0.0));
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] value "<< 0.25<<" -> "<<::eds::utils::valueToColor(0.25));
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] value "<< 0.5<<" -> "<<::eds::utils::valueToColor(0.5));
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] value "<< 0.75<<" -> "<<::eds::utils::valueToColor(0.75));
    BOOST_TEST_MESSAGE("[BOOST MESSAGE] value "<< 1.0<<" -> "<<::eds::utils::valueToColor(1.0));
}