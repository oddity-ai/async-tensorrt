use cpp::cpp;

cpp! {{
    #include <cstdint>
    #include <string>
}}

cpp! {{
    #include <cuda_runtime.h>
}}

cpp! {{
    #include <NvInfer.h>
    #include <NvOnnxParser.h>
}}

cpp! {{
    using namespace nvinfer1;
    using namespace nvonnxparser;
}}
