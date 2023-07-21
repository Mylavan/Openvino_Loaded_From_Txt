#include <iostream>
#include <Windows.h>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <ie/inference_engine.hpp>
#include <chrono>
#include <ctime>
#include <fstream>
using namespace InferenceEngine;



#define IM_SEGMENTATION_WIDTH            896    // default width of segmented frame
#define IM_SEGMENTATION_HEIGHT            640 // default height of segmented frame
#define IM_SEGMENTATION_CHANNEL            50  // max number of channel with segmented frame
using namespace cv;
using namespace std;
using namespace std::chrono;
int  PaddingTop = 0;
int PaddingBottom = 0;
int PaddingLeft = 0;
int PaddingRight = 0;
int Original_Input_Height = 0;
int Original_Input_Width = 0;
int outCroppedWidth = 0;
int outCroppedHeight = 0;
int outCroppedOriginY = 0;
int outCroppedOriginX = 0;
float ms = 0;


Mat loadImageandPreProcess(const std::string& filename, int sizeX = IM_SEGMENTATION_WIDTH, int sizeY = IM_SEGMENTATION_HEIGHT)
{
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cout << "No image found.";
    }
    int OriginalInputImageWidth = image.size().width;
    int OriginalInputImageHight = image.size().height;

    //closeup crop calculation
    cv::Rect rect = cv::boundingRect(image);

    outCroppedOriginX = rect.x;
    outCroppedOriginY = rect.y;
    outCroppedWidth = rect.width;
    outCroppedHeight = rect.height;

    cv::Mat croppedImage = image(rect);
    cv::Mat fCroppedImage;
    croppedImage.convertTo(fCroppedImage, CV_32FC1);


    //mean and standard deviation calculation
    cv::Scalar mean, stddev;
    cv::meanStdDev(fCroppedImage, mean, stddev);

    double dMean = mean[0];
    double dStdDev = stddev[0];



    //normalize image pxiel values using image mean & standard deviation
    // using formula : (img � img.mean() / img.std())
    fCroppedImage = (fCroppedImage - dMean) / (dStdDev + 1e-8);



    //old code 
    //cv::Mat fCroppedImageResized;
    //cv::resize(fCroppedImage, fCroppedImageResized, cv::Size(IM_SEGMENTATION_WIDTH, IM_SEGMENTATION_HEIGHT), cv::INTER_NEAREST);



    //New Code 
    int hinput = fCroppedImage.size().height;
    int winput = fCroppedImage.size().width;
    float  aspectRatio = 0;
    int Target_Height = IM_SEGMENTATION_HEIGHT;
    int Target_Width = IM_SEGMENTATION_WIDTH;
    int Resized_Height = 0;
    int Resized_Width = 0;
    //Equal 
    if (winput < hinput)
    {
        aspectRatio = (float)winput / hinput;
        std::cout << aspectRatio << std::endl;
        Resized_Height = Target_Height;
        Resized_Width = (float)aspectRatio * Resized_Height;
        if (Resized_Width > Target_Width)
        {
            Resized_Height = Resized_Height - ((Resized_Width - Target_Width) / aspectRatio);
            Resized_Width = aspectRatio * Resized_Height;
        }
    }
    else
    {
        aspectRatio = (float)hinput / winput;
        Resized_Width = Target_Width;
        Resized_Height = (float)(aspectRatio * Resized_Width);
        if (Resized_Height > Target_Height)
        {
            Resized_Width = Resized_Width - ((Resized_Height - Target_Height) / aspectRatio);
            Resized_Height = aspectRatio * Resized_Width;
        }
    }
    cv::Mat fCroppedImageResized;
    Original_Input_Height = OriginalInputImageHight;
    Original_Input_Width = OriginalInputImageWidth;
    cv::resize(fCroppedImage, fCroppedImageResized, cv::Size(Resized_Width, Resized_Height), cv::INTER_NEAREST);

    int DiffWidth = Target_Width - Resized_Width;
    int DiffHeight = Target_Height - Resized_Height;
    PaddingTop = DiffHeight / 2;
    PaddingBottom = DiffHeight / 2 + DiffHeight % 2;
    PaddingLeft = DiffWidth / 2;
    PaddingRight = DiffWidth / 2 + DiffWidth % 2;


    Sleep(3000);
    // Added sleep to show the differnce in async and normal execution , Async boosts the time taken ,the boosts can be seen when there is huge pre processing step , to replicate that state
    // have added the sleep .
    Mat PaddedImage;
    copyMakeBorder(fCroppedImageResized, PaddedImage, PaddingTop, PaddingBottom, PaddingLeft, PaddingRight, BORDER_CONSTANT, 0);

    //std::vector<float> vec;
    //int cn = 1;//RGBA , 4 channel
    //int iCount = 0;

    //const int inputNumChannel = 1;
    //const int inputH = IM_SEGMENTATION_HEIGHT;
    //const int inputW = IM_SEGMENTATION_WIDTH;

    //std::vector<float> vecR;

    //vecR.resize(inputH * inputW);


    //for (int i = 0; i < inputH; i++)
    //{
    //    for (int j = 0; j < inputW; j++)
    //    {
    //        float pixelValue = PaddedImage.at<float>(i, j);
    //        vecR[iCount] = pixelValue;
    //        iCount++;
    //    }
    //}
    //vector<float> input_tensor_values;
    //for (auto i = vecR.begin(); i != vecR.end(); ++i)
    //{
    //    input_tensor_values.push_back(*i);
    //}
    ////return input_tensor_values;
    return PaddedImage;
}

//int main()
//{
//    fstream ModelFile;
//
//    ModelFile.open("./Extra/Dependency/Models/ModelCurrentlyUsed.txt", ios::in); //open a file to perform read operation using file object
//    if (ModelFile.is_open()) { //checking whether the file is open
//    string modelname;
//    while (getline(ModelFile, modelname)) { //read data from file object and put it into string.
//        cout << modelname << "\n"; //print the data of the string
//    }
//    ModelFile.close(); //close the file object.
//}
//
//
//}
int main() {
    {
        Core ie;

        std::chrono::time_point<std::chrono::high_resolution_clock> ModelLoadstart, ModelLoadend, InferenceTimestart, InferenceTimeend, TotalTimeforAllFramesStart, TotalTimeforAllFramesEnd, TimeforOneFrameStart, TimeforOneFrameEnd, PreProcessstart, PreProcessend;
        fstream ModelFile;


        ModelFile.open("./Extra/Dependency/Models/ModelCurrentlyUsed.txt", ios::in); //open a file to perform read operation using file object
        string modelname;
        if (ModelFile.is_open()) { //checking whether the file is open

            while (getline(ModelFile, modelname)) { //read data from file object and put it into string.
                cout <<"Loaded Model ( from .txt file ) Name is ::  " <<modelname << "\n"; //print the data of the string
                break;
            }
            ModelFile.close(); //close the file object.
        }



        InferenceEngine::CNNNetwork network;
        string ModelPrefix = "./Extra/Dependency/Models/";
        string ModelPostfixbin = ".bin";
        string ModelPostfixXml = ".xml";
        string ModelPostfixOnnx = ".onnx";
        std::string model_path = ModelPrefix + modelname + ModelPostfixXml;
        std::string weights_path = ModelPrefix + modelname + ModelPostfixbin;
        const char* filexml = model_path.data();
        const char* filebin = weights_path.data();
       
       
        struct stat sb;     
        if ((stat(filexml, &sb) == 0 && !(sb.st_mode & S_IFDIR))&& (stat(filebin, &sb) == 0 && !(sb.st_mode & S_IFDIR)))
        {
            cout << "The path is valid!" << std::endl;
            network = ie.ReadNetwork(model_path, weights_path);
            //network = ie.ReadNetwork(onnx_path);  
        }
        else
        {
            cout << "The Path is invalid!" << std::endl;
            return 0;
        }    
      

        ModelLoadstart = std::chrono::high_resolution_clock::now();
        // Load the network into the Inference Engine
        ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");  //Most time taking step 
        ModelLoadend = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = ModelLoadend - ModelLoadstart;

        std::cout << "Model Loading Time  " << elapsed_seconds.count() << std::endl;
        std::cout << " This Happens Only once , So relax and take a seat " << std::endl;
        TotalTimeforAllFramesStart = std::chrono::high_resolution_clock::now();
        std::vector<InferRequest> inferRequests;
        inferRequests.push_back(executable_network.CreateInferRequest());
        vector<String> fn;
        string Image_Name;
        int counter = 0;
        glob("./Extra/input/*.png", fn);
        for (auto f : fn)
        {
            inferRequests.push_back(executable_network.CreateInferRequest());
            counter++;
            std::cout << "-------------------------------------------NEW FRAME PROCESSING-------------------------------------------" << std::endl;
            TimeforOneFrameStart = std::chrono::high_resolution_clock::now();
            // std::cout << f << std::endl;



            string str1 = "./Extra/input";



            // Find first occurrence of "geeks"
            size_t found = f.find(str1);
            /* std::cout << str1.size() << std::endl;*/
            string r = f.substr(str1.size() + 1, f.size());
            r.erase(r.length() - 4);
            // prints the result
            cout << "String is: " << r << std::endl;
            /* cout << "-------------------------------------" << std::endl;*/



            const std::string imageFile = f;



            // Set up the input and output blobs
            InputsDataMap input_info(network.getInputsInfo());
            const auto& input = input_info.begin()->second;
            input->setPrecision(Precision::FP32);
            input->setLayout(Layout::NCHW);



            OutputsDataMap output_info(network.getOutputsInfo());
            const auto& output = output_info.begin()->second;
            output->setPrecision(Precision::FP32);




            // Prepare the input data
            const size_t input_channels = input->getTensorDesc().getDims()[1];
            const size_t input_height = input->getTensorDesc().getDims()[2];
            const size_t input_width = input->getTensorDesc().getDims()[3];

            //Incase you want to check the input Dimension of the model
            /*std::cout << " input_channels :: " << input_channels << std::endl;
            std::cout << " input_height :: " << input_height << std::endl;
            std::cout << " input_width :: " << input_width << std::endl; */

            PreProcessstart = std::chrono::high_resolution_clock::now();
            Mat PaddedImage = loadImageandPreProcess(imageFile);
            // InferenceEngine::InferRequest infer_request = executable_network.CreateInferRequest();
            InferenceEngine::Blob::Ptr input_blob = inferRequests[counter].GetBlob(network.getInputsInfo().begin()->first);
            InferenceEngine::MemoryBlob::Ptr minput_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(input_blob);
            InferenceEngine::LockedMemory<void> minput_buffer = minput_blob->wmap();
            float* input_data = minput_buffer.as<float*>();
            const int inputH = IM_SEGMENTATION_HEIGHT;
            const int inputW = IM_SEGMENTATION_WIDTH;
            std::vector<float> vecR;

            vecR.resize(inputH * inputW);
            int iCount = 0;
            for (int i = 0; i < inputH; i++)
            {
                for (int j = 0; j < inputW; j++)
                {
                    float pixelValue = PaddedImage.at<float>(i, j);
                    input_data[iCount] = pixelValue;
                    iCount++;
                }
            }
            PreProcessend = std::chrono::high_resolution_clock::now();
            auto duration1 = duration_cast<milliseconds>(PreProcessend - PreProcessstart);
            std::cout << " PRE PROCESSING Time Taken (in milliseconds) :: " << duration1.count() << std::endl;

            // Infer
            InferenceTimestart = std::chrono::high_resolution_clock::now();
            inferRequests[counter].SetBlob(network.getInputsInfo().begin()->first, input_blob);
            inferRequests[counter].StartAsync();  //Starting the Async Function 
            InferenceTimeend = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = InferenceTimeend - InferenceTimestart;
            std::cout << " Inference Time Taken :: " << elapsed_seconds.count() << std::endl;
        }
        for (int i = 0; i < counter; i++) {
            inferRequests[i].Wait();

            {
                // Set up the input and output blobs
                InputsDataMap input_info(network.getInputsInfo());
                const auto& input = input_info.begin()->second;
                input->setPrecision(Precision::FP32);
                input->setLayout(Layout::NCHW);



                OutputsDataMap output_info(network.getOutputsInfo());
                const auto& output = output_info.begin()->second;
                output->setPrecision(Precision::FP32);

                // Get output tensor
                InferenceEngine::Blob::Ptr output_blob = inferRequests[i].GetBlob(network.getOutputsInfo().begin()->first);
                InferenceEngine::MemoryBlob::Ptr moutput_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob);
                InferenceEngine::LockedMemory<const void> moutput_buffer = moutput_blob->rmap();
                const float* output_data = moutput_buffer.as<float*>();

                const size_t output_channels = output->getTensorDesc().getDims()[1];
                const size_t output_height = output->getTensorDesc().getDims()[2];
                const size_t output_width = output->getTensorDesc().getDims()[3];

                //Incase you want to check the Output Dimensions
               /* std::cout << " output_channels :: " << output_channels << std::endl;
                std::cout << " output_height :: " << output_height << std::endl;
                std::cout << " output_width :: " << output_width << std::endl;*/

                const size_t output_size = output_channels * output_height * output_width;
                std::vector<float> output_vec(output_data, output_data + output_size);


                /////////////////////////////////////////////////////////////////

                int imgSize = IM_SEGMENTATION_WIDTH * IM_SEGMENTATION_HEIGHT;
                unsigned char frameWithMaxPixelValueIndex[IM_SEGMENTATION_WIDTH * IM_SEGMENTATION_HEIGHT];
                memset(frameWithMaxPixelValueIndex, 0, imgSize * sizeof(unsigned char));
                //#pragma omp parallel for
                for (int iPixelIndex = 0; iPixelIndex < imgSize; iPixelIndex++)
                {
                    float pixelValue = 0;
                    float pixelMaxValue = -INFINITY; //Initialzie max pixel value holder to negative INFINITY
                    int   channelIndexWithMaxPixelValue = 0;
                    for (int iChannelIndex = 0; iChannelIndex < 18; iChannelIndex++)
                    {
                        pixelValue = *(output_data + (iChannelIndex * imgSize + iPixelIndex));
                        if (pixelMaxValue < pixelValue)
                        {
                            pixelMaxValue = pixelValue;
                            channelIndexWithMaxPixelValue = iChannelIndex;
                        }
                    }



                    frameWithMaxPixelValueIndex[iPixelIndex] = channelIndexWithMaxPixelValue;
                }

                string r = "lol";
                cv::Mat cvframeWithMaxPixelValueIndex = cv::Mat(cv::Size(IM_SEGMENTATION_WIDTH, IM_SEGMENTATION_HEIGHT), CV_8UC1, frameWithMaxPixelValueIndex, cv::Mat::AUTO_STEP);
                cv::imwrite("./Extra/Output_Mask/" + r + ".png", cvframeWithMaxPixelValueIndex);


                //Remove applied padding  back to preprocessed cropped dimension
                cv::Mat preprocess_cvframeWithMaxPixelValueIndex;
                preprocess_cvframeWithMaxPixelValueIndex = cvframeWithMaxPixelValueIndex(Range(PaddingTop, IM_SEGMENTATION_HEIGHT - PaddingBottom), Range(PaddingLeft, IM_SEGMENTATION_WIDTH - PaddingRight));
                // cv::imwrite("frameWithMaxPixelValueIndex_after_preprocess_cropped_resize.png", preprocess_cvframeWithMaxPixelValueIndex);




                 //Resize back to Cropped Size 
                cv::Mat FinalResized;
                cv::resize(preprocess_cvframeWithMaxPixelValueIndex, FinalResized, cv::Size(outCroppedWidth, outCroppedHeight), 0, 0, cv::INTER_NEAREST);
                cv::Rect preprocess_cropp_rect;
                preprocess_cropp_rect.x = outCroppedOriginX;
                preprocess_cropp_rect.y = outCroppedOriginY;
                preprocess_cropp_rect.width = outCroppedWidth;
                preprocess_cropp_rect.height = outCroppedHeight;



                //Resize Back to original Input Dimensions 
                cv::Mat maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex = cv::Mat::zeros(cv::Size(Original_Input_Width, Original_Input_Height), CV_8UC1);
                FinalResized.copyTo(maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex(preprocess_cropp_rect));
                //  cv::imwrite("frameWithMaxPixelValueIndex_after_maskapplied_orginputimgsize_resize.png", maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex);
                cv::imwrite("./Extra/Final_Resized_mask/" + r + ".png", maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex);

                TimeforOneFrameEnd = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> outerelapsed_seconds = TimeforOneFrameEnd - TimeforOneFrameStart;
                std::cout << "********************* TIME TAKEN for 1 FRAME (seconds) :: " << outerelapsed_seconds.count() << "  ***************************" << std::endl;

            }
            TotalTimeforAllFramesEnd = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> outerelapsed_seconds = TotalTimeforAllFramesEnd - TotalTimeforAllFramesStart;
            std::cout << "********************* TOTAL TIME (seconds)   :: " << outerelapsed_seconds.count() << "  ***************************" << std::endl;

            /* catch (const InferenceEngine::Exception& ex) {
                 std::cerr << "OpenVINO exception caught: " << ex.what() << std::endl;
             }
             catch (const std::exception& ex) {
                 std::cerr << "Standard exception caught: " << ex.what() << std::endl;
             }
             catch (...) {
                 std::cerr << "Unknown exception caught" << std::endl;
             }*/
        }

    }
    // system("pause"); // <----------------------------------
    return 0;
}