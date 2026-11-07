#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "httplib.h"
#include "VoiceSynthesizer.h"
#include "WakeWordDetector.h"
#include "AudioCapture.h"
#include "whisper.h"

#include <nlohmann/json.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

using namespace std;

// Порог Wake Word: 0.18f — шорохи и вздохи во время речи Джарвиса больше не дают ложный __WAKE__
constexpr float WAKE_WORD_THRESHOLD = 0.18f;

#pragma pack(push, 1)
struct WavHeader {
    char riff_tag[4] = { 'R', 'I', 'F', 'F' };
    uint32_t riff_length = 0;
    char wave_tag[4] = { 'W', 'A', 'V', 'E' };
    char fmt_tag[4] = { 'f', 'm', 't', ' ' };
    uint32_t fmt_length = 16;
    uint16_t audio_format = 1;
    uint16_t num_channels = 1;
    uint32_t sample_rate = 22050;
    uint32_t byte_rate = 22050 * sizeof(int16_t);
    uint16_t block_align = sizeof(int16_t);
    uint16_t bits_per_sample = 16;
    char data_tag[4] = { 'd', 'a', 't', 'a' };
    uint32_t data_length = 0;
};
#pragma pack(pop)

static_assert(sizeof(WavHeader) == 44, "WavHeader must be exactly 44 bytes");

filesystem::path getExecutableDirectory() {
#ifdef _WIN32
    // Буфер в куче (vector), а не на стеке — убирает предупреждение C6262 (64 КБ стека)
    std::vector<wchar_t> executablePath(32768);
    const DWORD length = GetModuleFileNameW(
        nullptr,
        executablePath.data(),
        static_cast<DWORD>(executablePath.size())
    );
    if (length == 0) {
        throw runtime_error("Failed to determine executable directory");
    }
    return filesystem::path(executablePath.data(), executablePath.data() + length).parent_path();
#else
    return filesystem::current_path();
#endif
}

string floatToWav(const vector<float>& audioData, int sampleRate) {
    if (audioData.empty()) {
        return {};
    }
    if (sampleRate <= 0) {
        throw runtime_error("Invalid TTS sample rate");
    }

    const size_t audioBytes = audioData.size() * sizeof(int16_t);
    if (audioBytes > numeric_limits<uint32_t>::max()) {
        throw runtime_error("Generated audio is too large");
    }

    WavHeader header;
    header.sample_rate = static_cast<uint32_t>(sampleRate);
    header.byte_rate = header.sample_rate * header.num_channels * sizeof(int16_t);
    header.block_align = header.num_channels * sizeof(int16_t);
    header.data_length = static_cast<uint32_t>(audioBytes);
    header.riff_length = header.data_length + static_cast<uint32_t>(sizeof(WavHeader)) - 8;

    string wavBytes;
    wavBytes.resize(sizeof(WavHeader) + audioBytes);
    memcpy(wavBytes.data(), &header, sizeof(WavHeader));

    int16_t* pcmData = reinterpret_cast<int16_t*>(wavBytes.data() + sizeof(WavHeader));
    for (size_t i = 0; i < audioData.size(); ++i) {
        const float clamped = clamp(audioData[i], -1.0f, 1.0f);
        pcmData[i] = static_cast<int16_t>(lround(clamped * 32767.0f));
    }

    return wavBytes;
}

// Проверка наличия кириллических символов UTF-8 в строке (авто-определение русского текста)
bool containsCyrillic(const string& text) {
    for (size_t i = 0; i + 1 < text.size(); ++i) {
        const unsigned char b1 = static_cast<unsigned char>(text[i]);
        const unsigned char b2 = static_cast<unsigned char>(text[i + 1]);
        if ((b1 == 0xD0 || b1 == 0xD1) && (b2 >= 0x80 && b2 <= 0xBF)) {
            return true;
        }
    }
    return false;
}

// Отсекает "галлюцинации" Whisper на тишине/шуме: [BLANK_AUDIO], [Music], (laughs), *noise* и т.п.
bool isNoiseArtifact(const string& text) {
    const size_t first = text.find_first_not_of(" \t\r\n");
    if (first == string::npos) {
        return true; // одни пробелы
    }
    const size_t last = text.find_last_not_of(" \t\r\n");
    const string trimmed = text.substr(first, last - first + 1);

    if (trimmed.find("[BLANK_AUDIO]") != string::npos ||
        trimmed.find("[XBOX SOUND]") != string::npos ||
        trimmed.find("[BOOM]") != string::npos ||
        trimmed.find("???") != string::npos) {
        return true;
    }

    const char f = trimmed.front();
    const char l = trimmed.back();
    if ((f == '[' && l == ']') || (f == '(' && l == ')') || (f == '*' && l == '*')) {
        return true;
    }
    return false;
}

int getOptimalWhisperThreads() {
    const unsigned int hw = std::thread::hardware_concurrency();
    return static_cast<int>(std::min(8u, hw > 0u ? hw : 4u));
}

// Строгий детектор только из 3 разрешенных языков: en, ru, lt
// Заодно один раз вычисляет mel-спектрограмму в состоянии контекста, чтобы whisper_full не считал её повторно.
const char* detectAllowedLanguage(whisper_context* ctx, const vector<float>& pcmFloat, int nThreads, bool& melReady) {
    melReady = false;
    const int maxId = whisper_lang_max_id();
    vector<float> probs(maxId + 1, 0.0f);

    if (whisper_pcm_to_mel(ctx, pcmFloat.data(), static_cast<int>(pcmFloat.size()), nThreads) != 0) {
        return "en";
    }
    melReady = true;

    if (whisper_lang_auto_detect(ctx, 0, nThreads, probs.data()) < 0) {
        return "en";
    }

    const int idEn = whisper_lang_id("en");
    const int idRu = whisper_lang_id("ru");
    const int idLt = whisper_lang_id("lt");

    const float pEn = (idEn >= 0) ? probs[idEn] : 0.0f;
    const float pRu = (idRu >= 0) ? probs[idRu] : 0.0f;
    const float pLt = (idLt >= 0) ? probs[idLt] : 0.0f;

    const char* chosen = "en";
    float bestProb = pEn;

    if (pRu > bestProb) {
        bestProb = pRu;
        chosen = "ru";
    }
    if (pLt > bestProb) {
        bestProb = pLt;
        chosen = "lt";
    }

    cout << "[OMNI STT] 3-Lang Filter -> Selected: '" << chosen
        << "' (en=" << pEn << ", ru=" << pRu << ", lt=" << pLt << ")" << endl;

    return chosen;
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    whisper_context* whisperContext = nullptr;

    try {
        const filesystem::path executableDirectory = getExecutableDirectory();
        cout << "[Jarvis] Executable directory: " << executableDirectory.string() << endl;

        // 1. Порог Wake Word: 0.18f, чтобы шорохи во время ответа не давали ложный __WAKE__
        const filesystem::path modelsDir = executableDirectory / "models";
        auto wakeWordDetector = make_shared<WakeWordDetector>(modelsDir, WAKE_WORD_THRESHOLD);
        cout << "[Jarvis] Wake Word Detector initialized (threshold = " << WAKE_WORD_THRESHOLD << ")." << endl;

        const filesystem::path whisperModelPath = executableDirectory / "ggml-base.bin";
        const filesystem::path espeakDataPath = executableDirectory / "espeak-ng-data";

        if (!filesystem::exists(whisperModelPath)) {
            throw runtime_error("Whisper model not found: " + whisperModelPath.string());
        }
        if (!filesystem::exists(espeakDataPath)) {
            throw runtime_error("espeak-ng-data not found: " + espeakDataPath.string());
        }

        // ======================================
        // 2. Инициализация Мультиязычного Voice Pool (EN, RU, LT)
        // ======================================
        unordered_map<string, unique_ptr<VoiceSynthesizer>> voicePool;

        auto registerVoice = [&](const string& lang, const string& onnxFile) {
            const filesystem::path modelPath = modelsDir / onnxFile;
            const filesystem::path configPath = modelsDir / (onnxFile + ".json");

            if (filesystem::exists(modelPath) && filesystem::exists(configPath)) {
                cout << "[Jarvis TTS] Loading [" << lang << "] voice: " << onnxFile << "..." << endl;
                voicePool[lang] = make_unique<VoiceSynthesizer>(
                    modelPath.string(),
                    configPath.string(),
                    espeakDataPath.string()
                );
            }
            else {
                cerr << "[Jarvis TTS] WARNING: Voice model for [" << lang << "] not found at: "
                    << modelPath.string() << endl;
            }
            };

        registerVoice("ru", "ru_RU-denis-medium.onnx");
        registerVoice("lt", "lt_LT-reginute1-medium.onnx");
        registerVoice("en", "en_US-lessac-medium.onnx");

        if (voicePool.find("en") == voicePool.end()) {
            throw runtime_error("Primary English voice (en_US-lessac-medium.onnx) is required!");
        }

        // ======================================
        // 3. Загрузка Whisper
        // ======================================
        cout << "[Jarvis STT] Loading Whisper model..." << endl;
        whisper_context_params contextParams = whisper_context_default_params();
        whisperContext = whisper_init_from_file_with_params(
            whisperModelPath.string().c_str(),
            contextParams
        );

        if (whisperContext == nullptr) {
            throw runtime_error("Failed to load Whisper model");
        }
        cout << "[Jarvis STT] Whisper model loaded!" << endl;

        mutex whisperMutex;
        mutex ttsMutex;

        // Неблокирующий сигнал пробуждения в отдельном потоке
        auto on_wake_word_detected = []() {
            std::thread([]() {
                try {
                    nlohmann::json wakeJson = { {"text", "__WAKE__"} };
                    httplib::Client cli("http://127.0.0.1:8080");
                    cli.set_connection_timeout(1, 0);
                    cli.set_read_timeout(1, 0);
                    cli.Post("/api/jarvis/command", wakeJson.dump(), "application/json");
                    cout << "[OMNI] Instant WAKE signal delivered to Java." << endl;
                }
                catch (...) {}
                }).detach();
            };

        // Распознавание речи строго на 3 языках (EN / RU / LT)
        auto on_speech_detected = [&whisperContext, &whisperMutex](const vector<float>& pcmFloat) {
            if (pcmFloat.empty()) {
                return;
            }

            string transcription = "";

            {
                lock_guard<mutex> lock(whisperMutex);
                cout << "\n[OMNI] Processing Wake Word Command..." << endl;

                const int optimalThreads = getOptimalWhisperThreads();
                bool melReady = false;
                const char* lockedLang = detectAllowedLanguage(whisperContext, pcmFloat, optimalThreads, melReady);

                whisper_full_params whisperParams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                whisperParams.language = lockedLang;
                if (strcmp(lockedLang, "en") == 0) {
                    whisperParams.initial_prompt = "Stop, shut up, quiet, notepad, spring boot, jarvis, tutorial, browser, GrapheneOS, One UI.";
                }
                whisperParams.translate = false;
                whisperParams.no_context = true;
                whisperParams.single_segment = true;
                whisperParams.print_progress = false;
                whisperParams.print_realtime = false;
                whisperParams.print_timestamps = false;
                whisperParams.n_threads = optimalThreads;

                // Pass (nullptr, 0) when melReady is true so whisper_full reuses the mel spectrogram
                // already computed in detectAllowedLanguage instead of running whisper_pcm_to_mel twice.
                const int inferenceResult = whisper_full(
                    whisperContext,
                    whisperParams,
                    melReady ? nullptr : pcmFloat.data(),
                    melReady ? 0 : static_cast<int>(pcmFloat.size())
                );

                if (inferenceResult != 0) {
                    cerr << "[OMNI] Whisper inference failed." << endl;
                    return;
                }

                const int segmentCount = whisper_full_n_segments(whisperContext);
                for (int segmentIndex = 0; segmentIndex < segmentCount; ++segmentIndex) {
                    const char* segmentText = whisper_full_get_segment_text(whisperContext, segmentIndex);
                    if (segmentText != nullptr) {
                        transcription += segmentText;
                    }
                }
            }

            if (!transcription.empty()) {
                if (isNoiseArtifact(transcription)) {
                    cout << "[OMNI] Dropping silence/noise artifact: " << transcription << endl;
                    return;
                }

                cout << "[OMNI] Transcription: " << transcription << endl;

                try {
                    nlohmann::json json = { {"text", transcription} };
                    httplib::Client cli("http://127.0.0.1:8080");
                    cli.set_connection_timeout(2, 0);
                    cli.set_read_timeout(5, 0);

                    auto res = cli.Post("/api/jarvis/command", json.dump(), "application/json");
                    if (res && res->status == 200) {
                        cout << "[OMNI] Sent to Java successfully." << endl;
                    }
                    else if (res) {
                        cerr << "[OMNI] Java returned error HTTP code: " << res->status << endl;
                    }
                    else {
                        cerr << "[OMNI] No response from Java (httplib error code: "
                            << static_cast<int>(res.error()) << "). Java may still be processing the command." << endl;
                    }
                }
                catch (const std::exception& e) {
                    cerr << "[OMNI] C++ Exception during network call: " << e.what() << endl;
                }
            }
            };

        AudioCapture audio_capture(
            wakeWordDetector,
            on_speech_detected,
            on_wake_word_detected
        );

        if (!audio_capture.start()) {
            throw runtime_error("Failed to start continuous audio capture.");
        }

        httplib::Server server;

        server.Get("/health", [](const httplib::Request&, httplib::Response& response) {
            const nlohmann::json result = {
                {"status", "ok"},
                {"service", "jarvis-voice-cpp"}
            };
            response.set_content(result.dump(), "application/json; charset=UTF-8");
            });

        server.Post("/transcribe", [whisperContext, &whisperMutex](const httplib::Request& request, httplib::Response& response) {
            try {
                if (!request.form.has_file("file")) {
                    response.status = 400;
                    response.set_content(R"({"error":"No file uploaded"})", "application/json; charset=UTF-8");
                    return;
                }

                const auto& file = request.form.get_file("file");
                const string& wavAudio = file.content;

                if (wavAudio.size() <= sizeof(WavHeader)) {
                    response.status = 400;
                    response.set_content(R"({"error":"Audio file is empty"})", "application/json; charset=UTF-8");
                    return;
                }

                const size_t pcmBytes = wavAudio.size() - sizeof(WavHeader);
                const size_t pcmLength = pcmBytes / sizeof(int16_t);
                vector<float> pcmFloat(pcmLength);
                const char* pcmStart = wavAudio.data() + sizeof(WavHeader);

                for (size_t i = 0; i < pcmLength; ++i) {
                    int16_t sample = 0;
                    memcpy(&sample, pcmStart + i * sizeof(int16_t), sizeof(int16_t));
                    pcmFloat[i] = static_cast<float>(sample) / 32768.0f;
                }

                string transcription;
                {
                    lock_guard<mutex> lock(whisperMutex);
                    const int optimalThreads = getOptimalWhisperThreads();
                    bool melReady = false;
                    const char* lockedLang = detectAllowedLanguage(whisperContext, pcmFloat, optimalThreads, melReady);

                    whisper_full_params whisperParams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                    whisperParams.language = lockedLang;
                    whisperParams.translate = false;
                    whisperParams.no_context = true;
                    whisperParams.single_segment = true;
                    whisperParams.print_progress = false;
                    whisperParams.print_realtime = false;
                    whisperParams.print_timestamps = false;
                    whisperParams.n_threads = optimalThreads;

                    if (whisper_full(
                            whisperContext,
                            whisperParams,
                            melReady ? nullptr : pcmFloat.data(),
                            melReady ? 0 : static_cast<int>(pcmFloat.size())
                        ) != 0) {
                        throw runtime_error("Whisper inference failed");
                    }
                    const int segmentCount = whisper_full_n_segments(whisperContext);
                    for (int i = 0; i < segmentCount; ++i) {
                        const char* text = whisper_full_get_segment_text(whisperContext, i);
                        if (text != nullptr) transcription += text;
                    }
                }

                const nlohmann::json result = { {"text", transcription} };
                response.set_content(result.dump(), "application/json; charset=UTF-8");
            }
            catch (const exception&) {
                response.status = 500;
                response.set_content(R"({"error":"Speech recognition failed"})", "application/json; charset=UTF-8");
            }
            });

        server.Post("/tts", [&voicePool, &ttsMutex](const httplib::Request& request, httplib::Response& response) {
            lock_guard<mutex> lock(ttsMutex);
            cout << "\n[Jarvis TTS] Received request" << endl;

            try {
                const nlohmann::json requestJson = nlohmann::json::parse(request.body);
                const string text = requestJson.value("text", "");
                string lang = requestJson.value("language", "en");

                if (text.empty()) {
                    response.status = 400;
                    response.set_content(R"({"error":"Text is empty"})", "application/json; charset=UTF-8");
                    return;
                }

                // Нормализация кода языка до 2 символов
                if (lang.size() > 2) {
                    lang = lang.substr(0, 2);
                }

                // Железная страховка: если в тексте есть кириллица — всегда говорим голосом Дениса (ru)!
                if (containsCyrillic(text)) {
                    lang = "ru";
                }
                else if (lang != "en" && lang != "ru" && lang != "lt") {
                    lang = "en";
                }

                // Выбираем синтезатор из пула (если нужная модель не загружена — фоллбэк на английский)
                auto it = voicePool.find(lang);
                if (it == voicePool.end()) {
                    cout << "[Jarvis TTS] Voice for '" << lang << "' not loaded, falling back to 'en'." << endl;
                    it = voicePool.find("en");
                }

                cout << "[Jarvis TTS] Voice: [" << it->first << "] | Text: " << text << endl;
                const SynthesizedAudio audio = it->second->synthesize(text);
                const string wavPayload = floatToWav(audio.samples, audio.sample_rate);

                response.set_content(wavPayload, "audio/wav");
                cout << "[Jarvis TTS] Response: " << wavPayload.size() << " bytes" << endl;
            }
            catch (const exception& e) {
                cerr << "[Jarvis TTS] ERROR: " << e.what() << endl;
                response.status = 500;
                response.set_content(R"({"error":"TTS synthesis failed"})", "application/json; charset=UTF-8");
            }
            });

        cout << "\n[Jarvis Voice Engine] Ready!" << endl;
        cout << "[Jarvis Voice Engine] http://127.0.0.1:5000" << endl;

        server.listen("0.0.0.0", 5000);
        whisper_free(whisperContext);
        return 0;
    }
    catch (const exception& exception) {
        cerr << "\n[Jarvis Voice Engine] FATAL ERROR: " << exception.what() << endl;
        if (whisperContext != nullptr) {
            whisper_free(whisperContext);
        }
        return 1;
    }
}
