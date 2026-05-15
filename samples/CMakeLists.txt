set(OPENSOURCE_SAMPLES_LIST
     sampleYoloV5
     sampleResNet
     sampleYoloX
     sampleBert
     sampleTypedNetwork
     sampleCondition
     sampleYoloV3
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
add_subdirectory(common)
add_subdirectory(plugins)
foreach(SAMPLE_ITER ${OPENSOURCE_SAMPLES_LIST})
    add_subdirectory(${SAMPLE_ITER})
endforeach(SAMPLE_ITER)
