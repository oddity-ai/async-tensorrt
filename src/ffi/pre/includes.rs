use cpp::cpp;

cpp! {{
    #include <cstdint>
    #include <string>
    #include <cstdio>
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
