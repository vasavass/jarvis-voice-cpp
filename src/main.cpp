#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "httplib.h"
#include "whisper.h"

using namespace std;

// Экранирование специальных символов для корректного JSON.
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

int main() {
    cout << "[Jarvis] Loading Whisper model..." << endl;

    whisper_context_params contextParams =
        whisper_context_default_params();

    whisper_context* ctx =
        whisper_init_from_file_with_params(
            "ggml-base.bin",
            contextParams
        );

    if (ctx == nullptr) {
        cerr << "ERROR: Failed to load ggml-base.bin!" << endl;
        cerr << "Make sure the model file exists." << endl;
        return 1;
    }

    cout << "[Jarvis] Model loaded successfully!" << endl;
    cout << "[Jarvis Voice Engine] C++ Core Initializing..." << endl;

    httplib::Server server;

    // cpp-httplib может обрабатывать несколько запросов параллельно.
    // Один Whisper context защищаем через mutex.
    mutex whisperMutex;

    server.Post(
        "/transcribe",
        [ctx, &whisperMutex](
            const httplib::Request& request,
            httplib::Response& response
            ) {
                cout << "\n[!] Received POST request to /transcribe" << endl;

                if (!request.form.has_file("file")) {
                    cout << " - ERROR: No 'file' field found." << endl;

                    response.status = 400;
                    response.set_content(
                        R"({"error":"No file uploaded"})",
                        "application/json; charset=UTF-8"
                    );

                    return;
                }

                const auto& file = request.form.get_file("file");

                cout << " - Filename: " << file.filename << endl;
                cout << " - Content-Type: " << file.content_type << endl;
                cout << " - File size: "
                    << file.content.size()
                    << " bytes" << endl;

                const string& audio = file.content;

                // Проверяем, есть ли данные после WAV-заголовка.
                if (audio.size() <= 44) {
                    cout << " - ERROR: Audio file is empty." << endl;

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

                // Один PCM16 sample занимает 2 байта.
                const size_t pcm16Length =
                    (audio.size() - 44) / sizeof(int16_t);

                vector<float> pcmf32;
                pcmf32.resize(pcm16Length);

                // PCM16 (-32768...32767) → float (-1.0...1.0).
                for (size_t i = 0; i < pcm16Length; ++i) {
                    pcmf32[i] =
                        static_cast<float>(pcm16[i]) / 32768.0f;
                }

                cout << " - PCM16 samples: "
                    << pcm16Length << endl;

                cout << " - Float samples: "
                    << pcmf32.size() << endl;

                // Настраиваем Whisper.
                whisper_full_params whisperParams =
                    whisper_full_default_params(
                        WHISPER_SAMPLING_GREEDY
                    );

                whisperParams.language = "auto";
                whisperParams.translate = false;

                whisperParams.print_progress = false;
                whisperParams.print_realtime = false;
                whisperParams.print_timestamps = false;

                whisperParams.n_threads = 4;

                string resultText;

                // Блокируем Whisper context на время всего распознавания
                // и получения сегментов.
                {
                    lock_guard<mutex> lock(whisperMutex);

                    cout << " - Whisper inference started..." << endl;

                    const int whisperResult =
                        whisper_full(
                            ctx,
                            whisperParams,
                            pcmf32.data(),
                            static_cast<int>(pcmf32.size())
                        );

                    if (whisperResult != 0) {
                        cerr << " - ERROR: Whisper inference failed." << endl;

                        response.status = 500;
                        response.set_content(
                            R"({"error":"Whisper inference failed"})",
                            "application/json; charset=UTF-8"
                        );

                        return;
                    }

                    const int segmentCount =
                        whisper_full_n_segments(ctx);

                    cout << " - Segments: "
                        << segmentCount << endl;

                    for (int i = 0; i < segmentCount; ++i) {
                        const char* segmentText =
                            whisper_full_get_segment_text(ctx, i);

                        if (segmentText != nullptr) {
                            resultText += segmentText;
                        }
                    }
                }

                cout << " - Transcription: "
                    << resultText << endl;

                const string jsonResponse =
                    "{\"text\":\"" +
                    escapeJson(resultText) +
                    "\"}";

                response.set_content(
                    jsonResponse,
                    "application/json; charset=UTF-8"
                );
        }
    );

    // Пока оставляем временный TTS endpoint.
    server.Post(
        "/tts",
        [](const httplib::Request& request,
            httplib::Response& response) {

                cout << "\n[!] Received POST request to /tts "
                    << "(Dummy Endpoint)" << endl;

                response.set_content("", "audio/wav");
        }
    );

    cout << "[Jarvis Voice Engine] Server listening on "
        << "http://127.0.0.1:5000" << endl;

    const bool started =
        server.listen("0.0.0.0", 5000);

    if (!started) {
        cerr << "ERROR: Failed to start server on port 5000."
            << endl;

        whisper_free(ctx);
        return 1;
    }

    // Освобождаем модель после остановки сервера.
    whisper_free(ctx);

    return 0;
}