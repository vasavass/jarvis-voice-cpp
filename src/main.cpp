#include <iostream>
#include <string>
#include "httplib.h"
#include "whisper.h"

using namespace std;

int main() {
    httplib::Server svr;

    cout << "[Jarvis Voice Engine] C++ Core Initializing..." << endl;

    svr.Post("/transcribe",
        [](const httplib::Request& req, httplib::Response& res) {
            cout << "\n[!] Received POST request to /transcribe" << endl;

            if (req.form.has_file("file")) {
                const auto& file = req.form.get_file("file");

                cout << " - Filename: " << file.filename << endl;
                cout << " - Content-Type: " << file.content_type << endl;
                cout << " - File size: "
                    << file.content.size()
                    << " bytes" << endl;

                string jsonResponse =
                    R"({"text":"Привет, я C++ сервер. Я получил твое аудио!"})";

                res.set_content(
                    jsonResponse,
                    "application/json; charset=UTF-8"
                );
            }
            else {
                cout << " - ERROR: No 'file' field found." << endl;

                res.status = 400;
                res.set_content(
                    R"({"error":"No file uploaded"})",
                    "application/json"
                );
            }
        }
    );

    cout << "[Jarvis Voice Engine] Server listening on "
        << "http://localhost:5000" << endl;

    svr.listen("0.0.0.0", 5000);

    svr.Post("/tts", [](const httplib::Request& req, httplib::Response& res) {
        cout << "\n[!] Received POST request to /tts (Dummy Endpoint)" << endl;
        // Возвращаем пустую строку с типом audio/wav
        res.set_content("", "audio/wav");
        });


    return 0;
}