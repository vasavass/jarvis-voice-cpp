#pragma once

#include <mutex>
#include <string>
#include <vector>

#include <piper.h>

struct SynthesizedAudio {
    std::vector<float> samples;
    int sample_rate = 22050;
};

class VoiceSynthesizer {
private:
    piper_synthesizer* synthesizer_ = nullptr;
    std::mutex mutex_;

public:
    VoiceSynthesizer(
        const std::string& model_path,
        const std::string& config_path,
        const std::string& espeak_data_path
    );

    ~VoiceSynthesizer();

    VoiceSynthesizer(const VoiceSynthesizer&) = delete;
    VoiceSynthesizer& operator=(const VoiceSynthesizer&) = delete;

    SynthesizedAudio synthesize(const std::string& text);
};