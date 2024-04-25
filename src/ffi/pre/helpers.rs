use cpp::cpp;

cpp! {{
    // Simple wrapper function for all destroyable TensorRT classes. Their
    // destroy methods are all deprecated but still required to correctly
    // free resources so we need them anyway. This function will make sure
    // that any deprecation warnings are ignored.
    template<typename T>
    void destroy(T* destroyable) {
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        #if NV_TENSORRT_MAJOR > 9
        destroyable->~T();
        #else
        destroyable->destroy();
        #endif
        #pragma GCC diagnostic pop
    }
}}
