#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "miniaudio/miniaudio.h"
#include "WakeWordDetector.h"

class AudioCapture {
private:
    static constexpr ma_uint32 SAMPLE_RATE = 16000;
    static constexpr std::size_t CHUNK_SIZE = 1280; // 80 ms at 16 kHz

    static constexpr std::size_t RING_BUFFER_SIZE =
        static_cast<std::size_t>(SAMPLE_RATE) * 3;

    ma_device device{};

    std::vector<int16_t> ring_buffer;
    std::vector<int16_t> chunk_buffer;

    std::atomic<std::size_t> write_index{ 0 };
    std::atomic<std::size_t> read_index{ 0 };

    std::atomic<bool> is_running{ false };

    std::thread worker_thread;

    std::function<void(const std::vector<float>&)> on_speech_detected;
    std::function<void()> on_wake_word_detected;

    std::shared_ptr<WakeWordDetector> wake_word_detector;

    static void data_callback(
        ma_device* pDevice,
        void* pOutput,
        const void* pInput,
        ma_uint32 frameCount
    );

    void worker_loop();

public:
    explicit AudioCapture(
        std::shared_ptr<WakeWordDetector> detector,
        std::function<void(const std::vector<float>&)> speech_callback = {},
        std::function<void()> wake_callback = {}
    );

    ~AudioCapture();

    AudioCapture(const AudioCapture&) = delete;
    AudioCapture& operator=(const AudioCapture&) = delete;

    bool start();
    void stop();
};
