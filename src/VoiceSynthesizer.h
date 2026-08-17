#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <iostream>

class VoiceSynthesizer {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

public:
    VoiceSynthesizer(const std::wstring& model_path);
    std::vector<float> synthesize(
        const std::vector<int64_t>& phonemes
    );
};
