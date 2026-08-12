cmake_minimum_required(VERSION 3.15)

project(JarvisVoiceEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(whisper)

add_executable(jarvis_voice src/main.cpp)
if (WIN32)
    target_link_libraries(jarvis_voice ws2_32 crypt32 whisper)
endif()
