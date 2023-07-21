# Openvino_Async
Async() of Openvino allows multithreading of inference , In this repo it is explained how it is done

StartAsync() and Wait() are methods in OpenVINO C++ API that allow you to perform asynchronous inference.
StartAsync() is used to start an asynchronous inference request. This method returns immediately, allowing you to perform other tasks while the inference request is being executed in the background. Once the inference request is complete, you can use the Wait() method to wait for the results.
Wait() is used to wait for an asynchronous inference request to complete. This method blocks the current thread until the inference request is complete, at which point it returns the results of the inference request.
By using StartAsync() and Wait(), you can perform inference on multiple images concurrently, which can improve the throughput of the inference application. For example, you can start an inference request for each image using StartAsync(), and then wait for the results using Wait() after all the inference requests have been started. This allows you to perform other tasks while the inference requests are being executed in the background, and then retrieve the results once they are complete.

**Here the Models are chosen based on the txt file , the text file is present in ./Extra/Dependency/Models/ModelCurrentlyUsed.txt**
