#define MINIAUDIO_IMPLEMENTATION

#include "AudioCapture.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <utility>

AudioCapture::AudioCapture(
    std::shared_ptr<WakeWordDetector> detector,
    std::function<void(const std::vector<float>&)> speech_callback,
    std::function<void()> wake_callback
)
    : wake_word_detector(std::move(detector)),
    on_speech_detected(std::move(speech_callback)),
    on_wake_word_detected(std::move(wake_callback)) {

    ring_buffer.resize(RING_BUFFER_SIZE, 0);
    chunk_buffer.resize(CHUNK_SIZE, 0);
}

AudioCapture::~AudioCapture() {
    stop();
}

void AudioCapture::data_callback(
    ma_device* pDevice,
    void* pOutput,
    const void* pInput,
    ma_uint32 frameCount
) {
    (void)pOutput;

    if (pDevice == nullptr || pInput == nullptr) {
        return;
    }

    auto* capture = static_cast<AudioCapture*>(pDevice->pUserData);

    if (capture == nullptr) {
        return;
    }

    const auto* input_samples = static_cast<const int16_t*>(pInput);

    std::size_t local_write_index = capture->write_index.load(std::memory_order_relaxed);

    for (ma_uint32 i = 0; i < frameCount; ++i) {
        const std::size_t buffer_position = local_write_index % RING_BUFFER_SIZE;
        capture->ring_buffer[buffer_position] = input_samples[i];
        ++local_write_index;
    }

    capture->write_index.store(local_write_index, std::memory_order_release);
}

bool AudioCapture::start() {
    if (is_running.load(std::memory_order_acquire)) {
        return true;
    }

    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_capture);

    deviceConfig.capture.format = ma_format_s16;
    deviceConfig.capture.channels = 1;
    deviceConfig.sampleRate = SAMPLE_RATE;
    deviceConfig.dataCallback = data_callback;
    deviceConfig.pUserData = this;

    if (ma_device_init(nullptr, &deviceConfig, &device) != MA_SUCCESS) {
        std::cerr << "[AudioCapture] Failed to initialize capture device." << std::endl;
        return false;
    }

    if (ma_device_start(&device) != MA_SUCCESS) {
        std::cerr << "[AudioCapture] Failed to start capture device." << std::endl;
        ma_device_uninit(&device);
        return false;
    }

    is_running.store(true, std::memory_order_release);
    worker_thread = std::thread(&AudioCapture::worker_loop, this);

    std::cout << "[AudioCapture] Listening..." << std::endl;
    return true;
}

void AudioCapture::stop() {
    if (!is_running.load(std::memory_order_acquire)) {
        return;
    }

    is_running.store(false, std::memory_order_release);

    if (worker_thread.joinable()) {
        worker_thread.join();
    }

    ma_device_uninit(&device);
    std::cout << "[AudioCapture] Stopped." << std::endl;
}

float calculate_rms(const std::vector<int16_t>& chunk) {
    if (chunk.empty()) {
        return 0.0f;
    }

    double sum = 0.0;
    for (const int16_t sample : chunk) {
        const double sample_value = static_cast<double>(sample);
        sum += sample_value * sample_value;
    }

    const double mean = sum / static_cast<double>(chunk.size());
    return static_cast<float>(std::sqrt(mean));
}

void AudioCapture::worker_loop() {
    std::size_t local_read_index = 0;

    constexpr float RMS_THRESHOLD = 650.0f;             // Порог голоса
    constexpr int INITIAL_SILENCE_LIMIT = 15;           // ~1.2 сек ожидания, если после Hey Jarvis тишина
    constexpr int SPEECH_PAUSE_LIMIT = 22;              // ~1.76 сек паузы на раздумье во время фразы
    constexpr int MAX_RECORDING_FRAMES = 250;           // До 20 секунд на длинный вопрос!

    bool is_recording = false;
    int silence_counter = 0;
    int recording_frames = 0;
    int active_speech_frames = 0;

    // Pre-reserve 20 seconds of 16 kHz float audio once so recording never triggers reallocations
    std::vector<float> command_buffer;
    command_buffer.reserve(16000 * 20);

    // Reference pre-allocated chunk buffer (zero allocations in the 80ms hot loop)
    std::vector<int16_t>& chunk = chunk_buffer;

    while (is_running.load(std::memory_order_acquire)) {
        const std::size_t current_write_index = write_index.load(std::memory_order_acquire);
        const std::size_t available_samples = current_write_index - local_read_index;

        if (available_samples >= CHUNK_SIZE) {
            for (std::size_t i = 0; i < CHUNK_SIZE; ++i) {
                const std::size_t buffer_position = local_read_index % RING_BUFFER_SIZE;
                chunk[i] = ring_buffer[buffer_position];
                ++local_read_index;
            }

            read_index.store(local_read_index, std::memory_order_release);

            const float rms = calculate_rms(chunk);

            // ==========================================
            // IDLE: слушаем Wake Word ("Hey Jarvis")
            // ==========================================
            if (!is_recording) {
                if (wake_word_detector != nullptr && wake_word_detector->process(chunk)) {
                    wake_word_detector->reset();

                    std::cout << "\n[OMNI] Wake Word Detected! Interrupting playback & listening..." << std::endl;

                    // 1. Асинхронно глушим динамики в Java без задержки аудио-потока
                    if (on_wake_word_detected) {
                        on_wake_word_detected();
                    }

                    // 2. Короткая пауза 850 мс для гашения эха колонок
                    std::this_thread::sleep_for(std::chrono::milliseconds(850));
                    local_read_index = write_index.load(std::memory_order_acquire);
                    read_index.store(local_read_index, std::memory_order_release);

                    is_recording = true;
                    silence_counter = 0;
                    recording_frames = 0;
                    active_speech_frames = 0;
                    command_buffer.clear();
                }
            }
            // ==========================================
            // RECORDING: пишем команду пользователя
            // ==========================================
            else {
                for (const int16_t sample : chunk) {
                    command_buffer.push_back(static_cast<float>(sample) / 32768.0f);
                }

                ++recording_frames;

                if (rms < RMS_THRESHOLD) {
                    ++silence_counter;
                }
                else {
                    silence_counter = 0;
                    ++active_speech_frames;
                }

                // Если пользователь уже начал говорить, даем ему 1.76 сек на паузы между словами.
                // Если он просто сказал "Hey Jarvis" и молчит — завершаем через 1.2 сек.
                const int current_silence_limit =
                    (active_speech_frames > 2) ? SPEECH_PAUSE_LIMIT : INITIAL_SILENCE_LIMIT;

                if (silence_counter > current_silence_limit || recording_frames >= MAX_RECORDING_FRAMES) {
                    std::cout << "[VAD] End of speech detected (active frames: "
                        << active_speech_frames << "). Processing..." << std::endl;

                    if (active_speech_frames <= 2) {
                        std::cout << "[VAD] Only Wake Word spoken (Barge-In Stop). Skipping STT." << std::endl;
                    }
                    else if (on_speech_detected) {
                        on_speech_detected(command_buffer);
                    }

                    local_read_index = write_index.load(std::memory_order_acquire);
                    read_index.store(local_read_index, std::memory_order_release);

                    is_recording = false;
                    silence_counter = 0;
                    recording_frames = 0;
                    active_speech_frames = 0;
                }
            }
        }
        else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}
