cmake_minimum_required(VERSION 3.15)

project(JarvisVoiceEngine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==========================================
# Статическая сборка Whisper
# ==========================================

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)

include(FetchContent)

# ==========================================
# Whisper.cpp
# ==========================================

FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
    GIT_TAG        v1.9.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(whisper)

# ==========================================
# JSON (nlohmann)
# ==========================================

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.12.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# ==========================================
# ONNX Runtime 1.22
# ==========================================

set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIRECTORY "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIRECTORY}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime headers not found: ${ONNXRUNTIME_INCLUDE_DIRECTORY}")
endif()

if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
    message(FATAL_ERROR "onnxruntime.lib not found: ${ONNXRUNTIME_LIBRARY}")
endif()

if(NOT EXISTS "${ONNXRUNTIME_DLL}")
    message(FATAL_ERROR "onnxruntime.dll not found: ${ONNXRUNTIME_DLL}")
endif()

add_library(onnxruntime_runtime SHARED IMPORTED GLOBAL)
set_target_properties(
    onnxruntime_runtime
    PROPERTIES
    IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}"
    IMPORTED_LOCATION "${ONNXRUNTIME_DLL}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRECTORY}"
)

# ==========================================
# libpiper
# ==========================================

get_filename_component(
    PIPER_ROOT
    "${CMAKE_SOURCE_DIR}/../piper1-gpl/libpiper/install"
    ABSOLUTE
)

set(PIPER_INCLUDE_DIRECTORY "${PIPER_ROOT}/include")
set(PIPER_HEADER "${PIPER_INCLUDE_DIRECTORY}/piper.h")
set(PIPER_LIBRARY "${PIPER_ROOT}/lib/piper.lib")
set(PIPER_DLL "${PIPER_ROOT}/lib/piper.dll")
set(PIPER_ONNXRUNTIME_DLL "${ONNXRUNTIME_DLL}")
set(PIPER_PROVIDERS_DLL "${PIPER_ROOT}/lib/onnxruntime_providers_shared.dll")
set(ESPEAK_DATA_DIRECTORY "${PIPER_ROOT}/share/espeak-ng-data")

if(NOT EXISTS "${PIPER_HEADER}")
    message(FATAL_ERROR "piper.h not found: ${PIPER_HEADER}")
endif()

if(NOT EXISTS "${PIPER_LIBRARY}")
    message(FATAL_ERROR "piper.lib not found: ${PIPER_LIBRARY}")
endif()

if(NOT EXISTS "${PIPER_DLL}")
    message(FATAL_ERROR "piper.dll not found: ${PIPER_DLL}")
endif()

if(NOT EXISTS "${PIPER_ONNXRUNTIME_DLL}")
    message(FATAL_ERROR "ONNX Runtime DLL not found: ${PIPER_ONNXRUNTIME_DLL}")
endif()

if(NOT EXISTS "${ESPEAK_DATA_DIRECTORY}")
    message(FATAL_ERROR "espeak-ng-data not found: ${ESPEAK_DATA_DIRECTORY}")
endif()

add_library(piper_runtime SHARED IMPORTED GLOBAL)
set_target_properties(
    piper_runtime
    PROPERTIES
    IMPORTED_IMPLIB "${PIPER_LIBRARY}"
    IMPORTED_LOCATION "${PIPER_DLL}"
    INTERFACE_INCLUDE_DIRECTORIES "${PIPER_INCLUDE_DIRECTORY}"
)

# ==========================================
# Jarvis Voice Engine Executable
# ==========================================

add_executable(
    jarvis_voice
    src/main.cpp
    src/VoiceSynthesizer.cpp
    src/VoiceSynthesizer.h
    src/AudioCapture.cpp
    src/AudioCapture.h
    src/WakeWordDetector.cpp
    src/WakeWordDetector.h
)

target_include_directories(
    jarvis_voice
    PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/third_party"
)

target_link_libraries(
    jarvis_voice
    PRIVATE
    whisper
    piper_runtime
    onnxruntime_runtime
    nlohmann_json::nlohmann_json
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
# Post-Build: Копирование Piper и ONNX Runtime DLL
# ==========================================

add_custom_command(
    TARGET jarvis_voice
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PIPER_DLL}"
        "$<TARGET_FILE_DIR:jarvis_voice>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PIPER_ONNXRUNTIME_DLL}"
        "$<TARGET_FILE_DIR:jarvis_voice>"
    COMMENT "Copying Piper and ONNX Runtime DLLs"
    VERBATIM
)

if(EXISTS "${PIPER_PROVIDERS_DLL}")
    add_custom_command(
        TARGET jarvis_voice
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PIPER_PROVIDERS_DLL}"
            "$<TARGET_FILE_DIR:jarvis_voice>"
        COMMENT "Copying ONNX Runtime providers DLL"
        VERBATIM
    )
endif()

# ==========================================
# Post-Build: Копирование espeak-ng-data
# ==========================================

add_custom_command(
    TARGET jarvis_voice
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${ESPEAK_DATA_DIRECTORY}"
        "$<TARGET_FILE_DIR:jarvis_voice>/espeak-ng-data"
    COMMENT "Copying espeak-ng language data"
    VERBATIM
)

# ==========================================
# Post-Build: Копирование Whisper модели
# ==========================================

set(WHISPER_MODEL "${CMAKE_SOURCE_DIR}/ggml-base.bin")

if(EXISTS "${WHISPER_MODEL}")
    add_custom_command(
        TARGET jarvis_voice
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${WHISPER_MODEL}"
            "$<TARGET_FILE_DIR:jarvis_voice>"
        COMMENT "Copying Whisper model"
        VERBATIM
    )
else()
    message(WARNING "Whisper model not found: ${WHISPER_MODEL}")
endif()

# ==========================================
# Post-Build: Копирование всех моделей (TTS + WakeWord)
# ==========================================

set(MODELS_DIRECTORY "${CMAKE_SOURCE_DIR}/models")

if(EXISTS "${MODELS_DIRECTORY}")
    add_custom_command(
        TARGET jarvis_voice
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${MODELS_DIRECTORY}"
            "$<TARGET_FILE_DIR:jarvis_voice>/models"
        COMMENT "Copying Piper and openWakeWord models"
        VERBATIM
    )
else()
    message(WARNING "Models directory not found: ${MODELS_DIRECTORY}")
endif()