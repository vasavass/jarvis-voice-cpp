#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "httplib.h"
#include "VoiceSynthesizer.h"
#include "whisper.h"

using namespace std;

// ==========================================
// JSON escaping
// ==========================================

string escapeJson(const string& text) {
    string escaped;

    for (char symbol : text) {
        switch (symbol) {
        case '"':
            escaped += "\\\"";
            break;

        case '\\':
            escaped += "\\\\";
            break;

        case '\n':
            escaped += "\\n";
            break;

        case '\r':
            escaped += "\\r";
            break;

        case '\t':
            escaped += "\\t";
            break;

        default:
            escaped += symbol;
        }
    }

    return escaped;
}

// ==========================================
// WAV header
// ==========================================

#pragma pack(push, 1)

struct WavHeader {
    char riff_tag[4] = {
        'R', 'I', 'F', 'F'
    };

    uint32_t riff_length = 0;

    char wave_tag[4] = {
        'W', 'A', 'V', 'E'
    };

    char fmt_tag[4] = {
        'f', 'm', 't', ' '
    };

    uint32_t fmt_length = 16;
    uint16_t audio_format = 1;
    uint16_t num_channels = 1;

    uint32_t sample_rate = 22050;

    uint32_t byte_rate =
        22050 * sizeof(int16_t);

    uint16_t block_align =
        sizeof(int16_t);

    uint16_t bits_per_sample = 16;

    char data_tag[4] = {
        'd', 'a', 't', 'a'
    };

    uint32_t data_length = 0;
};

#pragma pack(pop)

static_assert(
    sizeof(WavHeader) == 44,
    "WAV header must be exactly 44 bytes"
    );

// ==========================================
// Float audio -> PCM16 WAV
// ==========================================

string floatToWav(
    const vector<float>& audio_data
) {
    if (audio_data.empty()) {
        return {};
    }

    WavHeader header;

    header.data_length = static_cast<uint32_t>(
        audio_data.size() * sizeof(int16_t)
        );

    header.riff_length =
        header.data_length +
        static_cast<uint32_t>(sizeof(WavHeader)) -
        8;

    string wav_bytes;

    wav_bytes.resize(
        sizeof(WavHeader) + header.data_length
    );

    // Записываем WAV-заголовок.
    memcpy(
        wav_bytes.data(),
        &header,
        sizeof(WavHeader)
    );

    // Float (-1.0...1.0) -> PCM16.
    for (size_t i = 0; i < audio_data.size(); ++i) {
        const float clamped_sample = clamp(
            audio_data[i],
            -1.0f,
            1.0f
        );

        const int16_t pcm_sample =
            static_cast<int16_t>(
                clamped_sample * 32767.0f
                );

        memcpy(
            wav_bytes.data() +
            sizeof(WavHeader) +
            i * sizeof(int16_t),

            &pcm_sample,
            sizeof(int16_t)
        );
    }

    return wav_bytes;
}

// ==========================================
// Main
// ==========================================

int main() {
    // ==========================================
    // Loading Piper ONNX
    // ==========================================

    const wstring tts_model_path =
        L"models/en_US-lessac-high.onnx";

    VoiceSynthesizer tts(tts_model_path);

    // ==========================================
    // Loading Whisper
    // ==========================================

    cout << "[Jarvis STT] Loading Whisper model..."
        << endl;

    whisper_context_params context_params =
        whisper_context_default_params();

    whisper_context* whisper_context =
        whisper_init_from_file_with_params(
            "ggml-base.bin",
            context_params
        );

    if (whisper_context == nullptr) {
        cerr << "[Jarvis STT] ERROR: "
            << "Failed to load ggml-base.bin!"
            << endl;

        return 1;
    }

    cout << "[Jarvis STT] Whisper model loaded!"
        << endl;

    // ==========================================
    // HTTP server
    // ==========================================

    httplib::Server server;

    mutex whisper_mutex;
    mutex tts_mutex;

    // ==========================================
    // POST /transcribe
    // ==========================================

    server.Post(
        "/transcribe",

        [whisper_context, &whisper_mutex](
            const httplib::Request& request,
            httplib::Response& response
            ) {
                cout << "\n[Jarvis STT] "
                    << "Received POST /transcribe"
                    << endl;

                if (!request.form.has_file("file")) {
                    cerr << "[Jarvis STT] "
                        << "No 'file' field found."
                        << endl;

                    response.status = 400;

                    response.set_content(
                        R"({"error":"No file uploaded"})",
                        "application/json; charset=UTF-8"
                    );

                    return;
                }

                const auto& file =
                    request.form.get_file("file");

                cout << "[Jarvis STT] Filename: "
                    << file.filename << endl;

                cout << "[Jarvis STT] Content-Type: "
                    << file.content_type << endl;

                cout << "[Jarvis STT] File size: "
                    << file.content.size()
                    << " bytes" << endl;

                const string& audio = file.content;

                if (audio.size() <= 44) {
                    cerr << "[Jarvis STT] "
                        << "Audio file is empty."
                        << endl;

                    response.status = 400;

                    response.set_content(
                        R"({"error":"Audio file is empty"})",
                        "application/json; charset=UTF-8"
                    );

                    return;
                }

                // Пропускаем 44 байта WAV-заголовка.
                const int16_t* pcm16 =
                    reinterpret_cast<const int16_t*>(
                        audio.data() + 44
                        );

                const size_t pcm16_length =
                    (audio.size() - 44) /
                    sizeof(int16_t);

                vector<float> pcmf32;
                pcmf32.resize(pcm16_length);

                for (size_t i = 0;
                    i < pcm16_length;
                    ++i) {

                    pcmf32[i] =
                        static_cast<float>(pcm16[i]) /
                        32768.0f;
                }

                cout << "[Jarvis STT] PCM samples: "
                    << pcmf32.size() << endl;

                whisper_full_params whisper_params =
                    whisper_full_default_params(
                        WHISPER_SAMPLING_GREEDY
                    );

                whisper_params.language = "auto";
                whisper_params.translate = false;

                whisper_params.print_progress = false;
                whisper_params.print_realtime = false;
                whisper_params.print_timestamps = false;

                whisper_params.n_threads = 4;

                string result_text;

                {
                    lock_guard<mutex> lock(
                        whisper_mutex
                    );

                    cout << "[Jarvis STT] "
                        << "Inference started..."
                        << endl;

                    const int whisper_result =
                        whisper_full(
                            whisper_context,
                            whisper_params,
                            pcmf32.data(),
                            static_cast<int>(
                                pcmf32.size()
                                )
                        );

                    if (whisper_result != 0) {
                        cerr << "[Jarvis STT] "
                            << "Inference failed."
                            << endl;

                        response.status = 500;

                        response.set_content(
                            R"({"error":"Whisper inference failed"})",
                            "application/json; charset=UTF-8"
                        );

                        return;
                    }

                    const int segment_count =
                        whisper_full_n_segments(
                            whisper_context
                        );

                    cout << "[Jarvis STT] Segments: "
                        << segment_count << endl;

                    for (int i = 0;
                        i < segment_count;
                        ++i) {

                        const char* segment_text =
                            whisper_full_get_segment_text(
                                whisper_context,
                                i
                            );

                        if (segment_text != nullptr) {
                            result_text += segment_text;
                        }
                    }
                }

                cout << "[Jarvis STT] Transcription: "
                    << result_text << endl;

                const string json_response =
                    "{\"text\":\"" +
                    escapeJson(result_text) +
                    "\"}";

                response.set_content(
                    json_response,
                    "application/json; charset=UTF-8"
                );
        }
    );

    // ==========================================
    // POST /tts
    // ==========================================

    server.Post(
        "/tts",

        [&tts, &tts_mutex](
            const httplib::Request& request,
            httplib::Response& response
            ) {
                cout << "\n[Jarvis TTS] "
                    << "Received POST /tts"
                    << endl;

                // Временные случайные ID фонем.
                // После подключения phonemizer здесь
                // будут настоящие фонемы из текста.
                const vector<int64_t> test_phonemes = {
                    14,
                    5,
                    23,
                    76,
                    2,
                    8,
                    19,
                    3
                };

                vector<float> audio_array;

                {
                    lock_guard<mutex> lock(tts_mutex);

                    audio_array =
                        tts.synthesize(test_phonemes);
                }

                if (audio_array.empty()) {
                    cerr << "[Jarvis TTS] "
                        << "Failed to generate audio."
                        << endl;

                    response.status = 500;

                    response.set_content(
                        R"({"error":"TTS inference failed"})",
                        "application/json; charset=UTF-8"
                    );

                    return;
                }

                string wav_payload =
                    floatToWav(audio_array);

                if (wav_payload.empty()) {
                    cerr << "[Jarvis TTS] "
                        << "Failed to create WAV."
                        << endl;

                    response.status = 500;

                    response.set_content(
                        R"({"error":"Failed to create WAV"})",
                        "application/json; charset=UTF-8"
                    );

                    return;
                }

                cout << "[Jarvis TTS] Audio samples: "
                    << audio_array.size() << endl;

                cout << "[Jarvis TTS] WAV size: "
                    << wav_payload.size()
                    << " bytes" << endl;

                response.set_content(
                    wav_payload,
                    "audio/wav"
                );
        }
    );

    // ==========================================
    // Start server
    // ==========================================

    cout << "[Jarvis Voice Engine] "
        << "Server listening on "
        << "http://127.0.0.1:5000"
        << endl;

    const bool started =
        server.listen("0.0.0.0", 5000);

    if (!started) {
        cerr << "[Jarvis Voice Engine] "
            << "Failed to start server on port 5000."
            << endl;

        whisper_free(whisper_context);

        return 1;
    }

    // Выполнится после остановки сервера.
    whisper_free(whisper_context);

    return 0;
}