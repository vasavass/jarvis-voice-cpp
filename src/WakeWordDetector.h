#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

class WakeWordDetector {
public:
    explicit WakeWordDetector(
        const std::filesystem::path& models_directory,
        float detection_threshold = 0.5f
    );

    ~WakeWordDetector() = default;

    WakeWordDetector(const WakeWordDetector&) = delete;
    WakeWordDetector& operator=(const WakeWordDetector&) = delete;

    bool is_ready() const noexcept;

    bool process(const std::vector<int16_t>& audio_frame);
    void reset() noexcept;

private:
    float detection_threshold;
    static constexpr size_t FRAME_SIZE = 1280;
    static constexpr size_t CONTEXT_FRAMES = 76;
    static constexpr size_t FEATURES_PER_FRAME = 32;
    static constexpr size_t EMBEDDING_FRAMES = 16;
    static constexpr size_t EMBEDDING_DIM = 96;

    Ort::Env environment;
    Ort::SessionOptions session_options;
    Ort::MemoryInfo memory_info;

    std::unique_ptr<Ort::Session> melspectrogram_session;
    std::unique_ptr<Ort::Session> embedding_session;
    std::unique_ptr<Ort::Session> wake_word_session;

    // Pre-allocated zero-allocation hot-loop buffers
    std::vector<float> float_audio;
    std::vector<float> features_buffer;
    std::vector<float> embeddings_buffer;

    // Cached ONNX input/output node names (initialized once in constructor)
    std::string mel_input_name;
    std::string mel_output_name;
    std::string embed_input_name;
    std::string embed_output_name;
    std::string classifier_input_name;
    std::string classifier_output_name;

    std::array<const char*, 1> mel_input_names{};
    std::array<const char*, 1> mel_output_names{};
    std::array<const char*, 1> embed_input_names{};
    std::array<const char*, 1> embed_output_names{};
    std::array<const char*, 1> classifier_input_names{};
    std::array<const char*, 1> classifier_output_names{};
};
