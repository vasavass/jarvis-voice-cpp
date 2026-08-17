cmake_minimum_required(VERSION 3.15)

project(JarvisVoiceEngine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# Whisper.cpp
# ==========================================

# Собираем Whisper и GGML статически.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)

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

# ==========================================
# ONNX Runtime
# ==========================================

set(
    ONNXRUNTIME_ROOT
    "${CMAKE_SOURCE_DIR}/onnxruntime"
)

set(
    ONNXRUNTIME_INCLUDE_DIR
    "${ONNXRUNTIME_ROOT}/include"
)

set(
    ONNXRUNTIME_LIBRARY
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
)

set(
    ONNXRUNTIME_DLL
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
)

set(
    ONNXRUNTIME_PROVIDERS_DLL
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
)

if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
    message(
        FATAL_ERROR
        "onnxruntime.lib not found: ${ONNXRUNTIME_LIBRARY}"
    )
endif()

if(NOT EXISTS "${ONNXRUNTIME_DLL}")
    message(
        FATAL_ERROR
        "onnxruntime.dll not found: ${ONNXRUNTIME_DLL}"
    )
endif()

# Описываем готовую ONNX Runtime DLL как импортированную библиотеку.
add_library(
    onnxruntime
    SHARED
    IMPORTED
    GLOBAL
)

set_target_properties(
    onnxruntime
    PROPERTIES

    IMPORTED_IMPLIB
    "${ONNXRUNTIME_LIBRARY}"

    IMPORTED_LOCATION
    "${ONNXRUNTIME_DLL}"

    INTERFACE_INCLUDE_DIRECTORIES
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

# ==========================================
# Jarvis Voice Engine
# ==========================================

add_executable(
    jarvis_voice

    src/main.cpp
    src/VoiceSynthesizer.cpp
    src/VoiceSynthesizer.h
)

target_link_libraries(
    jarvis_voice
    PRIVATE

    whisper
    onnxruntime
)

if(WIN32)
    target_link_libraries(
        jarvis_voice
        PRIVATE

        ws2_32
        crypt32
    )
endif()

# ==========================================
# Копирование обязательных файлов
# ==========================================

add_custom_command(
    TARGET jarvis_voice
    POST_BUILD

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/ggml-base.bin"
        "$<TARGET_FILE_DIR:jarvis_voice>"

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DLL}"
        "$<TARGET_FILE_DIR:jarvis_voice>"

    COMMENT
        "Copying Whisper model and ONNX Runtime DLL"

    VERBATIM
)

# Эта DLL присутствует не во всех CPU-сборках ONNX Runtime.
if(EXISTS "${ONNXRUNTIME_PROVIDERS_DLL}")
    add_custom_command(
        TARGET jarvis_voice
        POST_BUILD

        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_PROVIDERS_DLL}"
            "$<TARGET_FILE_DIR:jarvis_voice>"

        COMMENT
            "Copying ONNX Runtime providers DLL"

        VERBATIM
    )
endif()

# Копируем TTS-модели, только если папка существует.
if(EXISTS "${CMAKE_SOURCE_DIR}/models")
    add_custom_command(
        TARGET jarvis_voice
        POST_BUILD

        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/models"
            "$<TARGET_FILE_DIR:jarvis_voice>/models"

        COMMENT
            "Copying TTS models"

        VERBATIM
    )
else()
    message(
        WARNING
        "Models directory not found: ${CMAKE_SOURCE_DIR}/models"
    )
endif()