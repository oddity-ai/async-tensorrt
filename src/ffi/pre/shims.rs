use cpp::cpp;

cpp! {{
    #if NV_TENSORRT_MAJOR < 10
    class Dims64
    {
    public:
        static constexpr int32_t MAX_DIMS{8};
        int32_t nbDims;
        int64_t d[MAX_DIMS];
    };
    #endif
}}
