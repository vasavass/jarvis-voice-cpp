cmake_minimum_required(VERSION 3.15)

project(JarvisVoiceEngine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Встраиваем Whisper и GGML прямо в jarvis_voice.exe.
# Отдельные whisper.dll и ggml*.dll не понадобятся.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)

# Не собираем лишние примеры и тесты Whisper.
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)

include(FetchContent)

FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
    GIT_TAG v1.9.2
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(whisper)

add_executable(
    jarvis_voice
    src/main.cpp
)

target_link_libraries(
    jarvis_voice
    PRIVATE
    whisper
)

if(WIN32)
    target_link_libraries(
        jarvis_voice
        PRIVATE
        ws2_32
        crypt32
    )
endif()

# После сборки автоматически кладём модель рядом с jarvis_voice.exe.
add_custom_command(
    TARGET jarvis_voice
    POST_BUILD

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/ggml-base.bin"
        "$<TARGET_FILE_DIR:jarvis_voice>"

    COMMENT "Copying ggml-base.bin next to jarvis_voice.exe"
    VERBATIM
)