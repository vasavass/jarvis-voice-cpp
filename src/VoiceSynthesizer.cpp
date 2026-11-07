#include "VoiceSynthesizer.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>

namespace {

    // Defensive normalization & padding for short strings (< 15 chars)
    // so espeak-ng / libpiper VITS never return 0 audio samples on short commands like "Done, sir." or "Yes, sir."
    std::string prepare_tts_text(const std::string& raw_text) {
        const auto first = raw_text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return {};
        }
        const auto last = raw_text.find_last_not_of(" \t\r\n");
        std::string processed = raw_text.substr(first, last - first + 1);

        const char back = processed.back();
        if (back != '.' && back != '!' && back != '?') {
            processed.push_back('.');
        }

        if (processed.size() < 15) {
            // Add trailing clause/pause padding for short utterances so VITS duration predictor
            // and STFT windowing always have sufficient phoneme frames.
            processed += " ";
        }

        return processed;
    }

} // namespace

VoiceSynthesizer::VoiceSynthesizer(
    const std::string& model_path,
    const std::string& config_path,
    const std::string& espeak_data_path
) {
    std::cout
        << "[Jarvis TTS] Initializing libpiper..."
        << std::endl;

    synthesizer_ = piper_create(
        model_path.c_str(),
        config_path.c_str(),
        espeak_data_path.c_str()
    );

    if (synthesizer_ == nullptr) {
        throw std::runtime_error(
            "Failed to create Piper synthesizer"
        );
    }

    std::cout
        << "[Jarvis TTS] Piper version: "
        << piper_version()
        << std::endl;

    std::cout
        << "[Jarvis TTS] Voice model loaded successfully!"
        << std::endl;
}

VoiceSynthesizer::~VoiceSynthesizer() {
    if (synthesizer_ != nullptr) {
        piper_free(synthesizer_);
        synthesizer_ = nullptr;
    }
}

SynthesizedAudio VoiceSynthesizer::synthesize(
    const std::string& text
) {
    const std::string prepared_text = prepare_tts_text(text);
    if (prepared_text.empty()) {
        return {};
    }

    // Один synthesizer нельзя одновременно использовать
    // из нескольких HTTP-потоков.
    std::lock_guard<std::mutex> lock(mutex_);

    piper_synthesize_options options =
        piper_default_synthesize_options(synthesizer_);

    options.length_scale = 1.0f;
    options.noise_scale = 0.667f;
    options.noise_w_scale = 0.8f;

    std::cout
        << "[Jarvis TTS] Starting synthesis: "
        << prepared_text
        << std::endl;

    auto run_piper_pass = [&](const std::string& input_text, SynthesizedAudio& out) {
        const int start_result = piper_synthesize_start(
            synthesizer_,
            input_text.c_str(),
            &options
        );

        if (start_result != PIPER_OK) {
            throw std::runtime_error(
                "piper_synthesize_start failed with code " +
                std::to_string(start_result)
            );
        }

        while (true) {
            piper_audio_chunk chunk{};

            const int chunk_result =
                piper_synthesize_next(
                    synthesizer_,
                    &chunk
                );

            // NOTE: libpiper returns PIPER_DONE on the final audio chunk (when chunk.is_last == true),
            // which for single-sentence phrases ("Done, sir.", "Yes, sir.") is the very first chunk!
            // Always harvest chunk.samples before breaking on PIPER_DONE / chunk.is_last.
            if (chunk_result != PIPER_OK && chunk_result != PIPER_DONE) {
                throw std::runtime_error(
                    "piper_synthesize_next failed with code " +
                    std::to_string(chunk_result)
                );
            }

            if (chunk.sample_rate > 0) {
                out.sample_rate = chunk.sample_rate;
            }

            if (chunk.samples != nullptr &&
                chunk.num_samples > 0) {

                out.samples.insert(
                    out.samples.end(),
                    chunk.samples,
                    chunk.samples + chunk.num_samples
                );
            }

            if (chunk.is_last || chunk_result == PIPER_DONE) {
                break;
            }
        }
    };

    SynthesizedAudio result;
    run_piper_pass(prepared_text, result);

    // Defensive fallback if espeak-ng produced 0 phoneme frames on a very short/unusual string
    if (result.samples.empty()) {
        const std::string fallback_text = prepared_text + " ... .";
        run_piper_pass(fallback_text, result);
    }

    // Ensure short phrases never crash with empty audio even on punctuation-only inputs
    if (result.samples.empty()) {
        const std::size_t silent_samples = static_cast<std::size_t>(
            (result.sample_rate > 0 ? result.sample_rate : 22050) / 10
        );
        result.samples.assign(silent_samples, 0.0f);
    }
    else if (prepared_text.size() < 15) {
        // Append 40ms of clean trailing silence pad for short phrases to prevent audio clipping
        const std::size_t pad_samples = static_cast<std::size_t>(result.sample_rate / 25);
        result.samples.insert(result.samples.end(), pad_samples, 0.0f);
    }

    std::cout
        << "[Jarvis TTS] Audio generated. Samples: "
        << result.samples.size()
        << ", sample rate: "
        << result.sample_rate
        << std::endl;

    return result;
}
