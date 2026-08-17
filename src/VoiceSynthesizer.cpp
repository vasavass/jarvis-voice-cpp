#include "VoiceSynthesizer.h"

#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

VoiceSynthesizer::VoiceSynthesizer(
    const std::wstring& model_path
)
    : env(
        ORT_LOGGING_LEVEL_WARNING,
        "JarvisTTS"
    ),
    session(nullptr),
    memory_info(
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        )
    )
{
    std::cout
        << "[Jarvis TTS] Initializing ONNX Environment..."
        << std::endl;

    Ort::SessionOptions session_options;

    session_options.SetIntraOpNumThreads(4);

    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    try {
        session = Ort::Session(
            env,
            model_path.c_str(),
            session_options
        );

        std::cout
            << "[Jarvis TTS] ONNX Model loaded successfully!"
            << std::endl;
    }
    catch (const Ort::Exception& exception) {
        std::cerr
            << "[Jarvis TTS] ERROR: Failed to load ONNX model: "
            << exception.what()
            << std::endl;

        throw;
    }
}

// Этот метод находится СНАРУЖИ конструктора.
std::vector<float> VoiceSynthesizer::synthesize(
    const std::vector<int64_t>& phonemes
) {
    if (phonemes.empty()) {
        return {};
    }

    // ==========================================
    // ЭТАП А: Подготовка данных
    // ==========================================

    std::vector<int64_t> phoneme_lengths = {
        static_cast<int64_t>(phonemes.size())
    };

    std::vector<float> scales = {
        0.667f, // noise_scale
        1.0f,   // length_scale
        0.8f    // noise_width
    };

    // ==========================================
    // ЭТАП Б: Формы тензоров
    // ==========================================

    std::vector<int64_t> phonemes_shape = {
        1,
        static_cast<int64_t>(phonemes.size())
    };

    std::vector<int64_t> phoneme_lengths_shape = {
        1
    };

    std::vector<int64_t> scales_shape = {
        3
    };

    // CreateTensor требует обычный указатель,
    // хотя ONNX Runtime не изменяет входной массив.
    int64_t* phonemes_data =
        const_cast<int64_t*>(phonemes.data());

    // ==========================================
    // ЭТАП В: Создание тензоров
    // ==========================================

    Ort::Value phonemes_tensor =
        Ort::Value::CreateTensor<int64_t>(
            memory_info,
            phonemes_data,
            phonemes.size(),
            phonemes_shape.data(),
            phonemes_shape.size()
        );

    Ort::Value lengths_tensor =
        Ort::Value::CreateTensor<int64_t>(
            memory_info,
            phoneme_lengths.data(),
            phoneme_lengths.size(),
            phoneme_lengths_shape.data(),
            phoneme_lengths_shape.size()
        );

    Ort::Value scales_tensor =
        Ort::Value::CreateTensor<float>(
            memory_info,
            scales.data(),
            scales.size(),
            scales_shape.data(),
            scales_shape.size()
        );

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(3);

    input_tensors.push_back(
        std::move(phonemes_tensor)
    );

    input_tensors.push_back(
        std::move(lengths_tensor)
    );

    input_tensors.push_back(
        std::move(scales_tensor)
    );

    // ==========================================
    // ЭТАП Г: Запуск модели Piper
    // ==========================================

    const char* input_names[] = {
        "input",
        "input_lengths",
        "scales"
    };

    const char* output_names[] = {
        "output"
    };

    std::cout
        << "[Jarvis TTS] Starting inference. Phonemes: "
        << phonemes.size()
        << std::endl;

    std::vector<Ort::Value> output_tensors;

    try {
        output_tensors = session.Run(
            Ort::RunOptions{ nullptr },
            input_names,
            input_tensors.data(),
            input_tensors.size(),
            output_names,
            1
        );
    }
    catch (const Ort::Exception& exception) {
        std::cerr
            << "[Jarvis TTS] Inference failed: "
            << exception.what()
            << std::endl;

        return {};
    }

    // ==========================================
    // ЭТАП Д: Извлечение звука
    // ==========================================

    if (output_tensors.empty()) {
        std::cerr
            << "[Jarvis TTS] Model returned no outputs."
            << std::endl;

        return {};
    }

    if (!output_tensors[0].IsTensor()) {
        std::cerr
            << "[Jarvis TTS] Output is not a tensor."
            << std::endl;

        return {};
    }

    float* output_data =
        output_tensors[0]
        .GetTensorMutableData<float>();

    const size_t output_size =
        output_tensors[0]
        .GetTensorTypeAndShapeInfo()
        .GetElementCount();

    if (output_data == nullptr || output_size == 0) {
        std::cerr
            << "[Jarvis TTS] Model returned empty audio."
            << std::endl;

        return {};
    }

    std::vector<float> audio_data(
        output_data,
        output_data + output_size
    );

    std::cout
        << "[Jarvis TTS] Audio generated. Samples: "
        << audio_data.size()
        << std::endl;

    return audio_data;
}