#include "WakeWordDetector.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

    std::filesystem::path require_model(
        const std::filesystem::path& directory,
        const std::string& filename
    ) {
        const std::filesystem::path model_path =
            directory / filename;

        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error(
                "Wake Word model not found: " +
                model_path.string()
            );
        }

        return model_path;
    }

} // namespace

WakeWordDetector::WakeWordDetector(
    const std::filesystem::path& models_directory,
    float detection_threshold
)
    : detection_threshold(detection_threshold),
    environment(
        ORT_LOGGING_LEVEL_WARNING,
        "OMNI-WakeWord"
    ),
    memory_info(
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        )
    ) {

    if (detection_threshold < 0.0f ||
        detection_threshold > 1.0f) {

        throw std::invalid_argument(
            "Wake Word threshold must be between 0.0 and 1.0"
        );
    }

    // Pre-allocate audio conversion buffer once (zero heap allocations in 80ms loop)
    float_audio.resize(FRAME_SIZE, 0.0f);

    // Заполняем первый буфер нулями
    features_buffer.resize(
        CONTEXT_FRAMES * FEATURES_PER_FRAME,
        0.0f
    );

    // Заполняем второй буфер нулями
    embeddings_buffer.resize(
        EMBEDDING_FRAMES * EMBEDDING_DIM,
        0.0f
    );

    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);

    session_options.SetExecutionMode(
        ExecutionMode::ORT_SEQUENTIAL
    );

    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    const std::filesystem::path melspectrogram_path =
        require_model(
            models_directory,
            "melspectrogram.onnx"
        );

    const std::filesystem::path embedding_path =
        require_model(
            models_directory,
            "embedding_model.onnx"
        );

    const std::filesystem::path wake_word_path =
        require_model(
            models_directory,
            "hey_jarvis_v0.1.onnx"
        );

    std::cout
        << "[WakeWord] Loading melspectrogram model..."
        << std::endl;

    melspectrogram_session =
        std::make_unique<Ort::Session>(
            environment,
            melspectrogram_path.c_str(),
            session_options
        );

    std::cout
        << "[WakeWord] Loading embedding model..."
        << std::endl;

    embedding_session =
        std::make_unique<Ort::Session>(
            environment,
            embedding_path.c_str(),
            session_options
        );

    std::cout
        << "[WakeWord] Loading Hey Jarvis model..."
        << std::endl;

    wake_word_session =
        std::make_unique<Ort::Session>(
            environment,
            wake_word_path.c_str(),
            session_options
        );

    // Cache ONNX input/output node names once at startup
    Ort::AllocatorWithDefaultOptions allocator;

    mel_input_name = melspectrogram_session->GetInputNameAllocated(0, allocator).get();
    mel_output_name = melspectrogram_session->GetOutputNameAllocated(0, allocator).get();
    mel_input_names[0] = mel_input_name.c_str();
    mel_output_names[0] = mel_output_name.c_str();

    embed_input_name = embedding_session->GetInputNameAllocated(0, allocator).get();
    embed_output_name = embedding_session->GetOutputNameAllocated(0, allocator).get();
    embed_input_names[0] = embed_input_name.c_str();
    embed_output_names[0] = embed_output_name.c_str();

    classifier_input_name = wake_word_session->GetInputNameAllocated(0, allocator).get();
    classifier_output_name = wake_word_session->GetOutputNameAllocated(0, allocator).get();
    classifier_input_names[0] = classifier_input_name.c_str();
    classifier_output_names[0] = classifier_output_name.c_str();

    std::cout
        << "[WakeWord] All ONNX models loaded successfully!"
        << std::endl;
}

bool WakeWordDetector::is_ready() const noexcept {
    return melspectrogram_session != nullptr &&
        embedding_session != nullptr &&
        wake_word_session != nullptr;
}

bool WakeWordDetector::process(
    const std::vector<int16_t>& audio_frame
) {
    // openWakeWord требует ровно 1280 сэмплов (80 мс звука)
    if (audio_frame.size() != FRAME_SIZE) {
        return false;
    }

    try {
        // ==========================================
        // STAGE 0: Нормализация (Zero-Allocation)
        // ==========================================
        for (size_t i = 0; i < FRAME_SIZE; ++i) {
            float_audio[i] = static_cast<float>(audio_frame[i]);
        }

        // ==========================================
        // STAGE 1: Melspectrogram (1280 сэмплов -> ровно 5 фреймов по 32 признака)
        // ==========================================
        static constexpr std::array<int64_t, 2> MEL_INPUT_SHAPE = {
            1, static_cast<int64_t>(FRAME_SIZE)
        };

        Ort::Value mel_input_tensor =
            Ort::Value::CreateTensor<float>(
                memory_info,
                float_audio.data(),
                float_audio.size(),
                MEL_INPUT_SHAPE.data(),
                MEL_INPUT_SHAPE.size()
            );

        auto mel_output_tensors =
            melspectrogram_session->Run(
                Ort::RunOptions{ nullptr },
                mel_input_names.data(),
                &mel_input_tensor,
                1,
                mel_output_names.data(),
                1
            );

        // ==========================================
        // STAGE 2: Embedding
        // ==========================================
        float* mel_out_data =
            mel_output_tensors[0]
            .GetTensorMutableData<float>();

        // 1280 сэмплов генерируют ровно 5 mel-фреймов: 5 * 32 = 160 float значений
        constexpr size_t NEW_MEL_FLOATS = 5 * FEATURES_PER_FRAME;

        // Сдвигаем историю влево на 160 элементов (5 фреймов)
        std::copy(
            features_buffer.begin() + NEW_MEL_FLOATS,
            features_buffer.end(),
            features_buffer.begin()
        );

        // Нормализуем и дописываем 5 новых фреймов в хвост буфера
        const size_t tail_offset =
            features_buffer.size() - NEW_MEL_FLOATS;

        for (size_t i = 0; i < NEW_MEL_FLOATS; ++i) {
            features_buffer[tail_offset + i] =
                (mel_out_data[i] / 10.0f) + 2.0f;
        }

        // Embedding получает полный временной контекст [1, 76, 32, 1]
        static constexpr std::array<int64_t, 4> EMBED_INPUT_SHAPE = {
            1,
            static_cast<int64_t>(CONTEXT_FRAMES),
            static_cast<int64_t>(FEATURES_PER_FRAME),
            1
        };

        Ort::Value embed_input_tensor =
            Ort::Value::CreateTensor<float>(
                memory_info,
                features_buffer.data(),
                features_buffer.size(),
                EMBED_INPUT_SHAPE.data(),
                EMBED_INPUT_SHAPE.size()
            );

        auto embed_output_tensors =
            embedding_session->Run(
                Ort::RunOptions{ nullptr },
                embed_input_names.data(),
                &embed_input_tensor,
                1,
                embed_output_names.data(),
                1
            );

        // ==========================================
        // STAGE 3: Wake Word Classifier
        // ==========================================
        float* embed_out_data =
            embed_output_tensors[0]
            .GetTensorMutableData<float>();

        // Сдвигаем историю эмбеддингов влево на 96 элементов
        std::copy(
            embeddings_buffer.begin() + EMBEDDING_DIM,
            embeddings_buffer.end(),
            embeddings_buffer.begin()
        );

        // Дописываем новый эмбеддинг (96 признаков) в конец буфера
        std::copy(
            embed_out_data,
            embed_out_data + EMBEDDING_DIM,
            embeddings_buffer.end() - EMBEDDING_DIM
        );

        static constexpr std::array<int64_t, 3> CLASSIFIER_INPUT_SHAPE = {
            1,
            static_cast<int64_t>(EMBEDDING_FRAMES),
            static_cast<int64_t>(EMBEDDING_DIM)
        };

        Ort::Value classifier_input_tensor =
            Ort::Value::CreateTensor<float>(
                memory_info,
                embeddings_buffer.data(),
                embeddings_buffer.size(),
                CLASSIFIER_INPUT_SHAPE.data(),
                CLASSIFIER_INPUT_SHAPE.size()
            );

        auto classifier_output_tensors =
            wake_word_session->Run(
                Ort::RunOptions{ nullptr },
                classifier_input_names.data(),
                &classifier_input_tensor,
                1,
                classifier_output_names.data(),
                1
            );

        // ==========================================
        // STAGE 4: Probability
        // ==========================================
        float probability =
            classifier_output_tensors[0]
            .GetTensorMutableData<float>()[0];

        // Диагностика: выводим скор в консоль при минимальной активности
        if (probability > 0.01f) {
            std::cout
                << "[WakeWord] Score: "
                << probability
                << std::endl;
        }

        return probability >= detection_threshold;

    }
    catch (const Ort::Exception& e) {
        std::cerr
            << "\n[WakeWord ONNX ERROR]: "
            << e.what()
            << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr
            << "\n[WakeWord STD ERROR]: "
            << e.what()
            << std::endl;
        return false;
    }
}

void WakeWordDetector::reset() noexcept {
    std::fill(float_audio.begin(), float_audio.end(), 0.0f);
    std::fill(features_buffer.begin(), features_buffer.end(), 0.0f);
    std::fill(embeddings_buffer.begin(), embeddings_buffer.end(), 0.0f);
}
