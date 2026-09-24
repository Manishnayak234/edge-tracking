#pragma once

// Minimal non-blocking serial link for the Arduino (raw 8N1, 115200 baud).
// Only what servo_track needs: open, write single-character commands, drain replies.

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

namespace edge_tracking::apps {

class SerialPort {
public:
    SerialPort() = default;
    ~SerialPort() { close(); }
    SerialPort(const SerialPort&) = delete;
    SerialPort& operator=(const SerialPort&) = delete;

    // Opens the port at 115200 8N1 in raw mode. Note that opening it toggles DTR, which
    // resets an Uno/Nano into its bootloader: wait ~2 s before the first command.
    bool open(const std::string& device, std::string* error = nullptr) {
        close();
        fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0) return fail(error, device + ": " + std::strerror(errno));

        termios tio{};
        if (tcgetattr(fd_, &tio) != 0) return fail(error, std::string("tcgetattr: ") + std::strerror(errno));
        cfmakeraw(&tio);
        cfsetispeed(&tio, B115200);
        cfsetospeed(&tio, B115200);
        tio.c_cflag |= CLOCAL | CREAD;  // ignore modem lines, enable the receiver
        tio.c_cflag &= ~CRTSCTS;        // no hardware flow control
        tio.c_cc[VMIN] = 0;             // reads return whatever is there, right away
        tio.c_cc[VTIME] = 0;
        if (tcsetattr(fd_, TCSANOW, &tio) != 0) return fail(error, std::string("tcsetattr: ") + std::strerror(errno));
        tcflush(fd_, TCIOFLUSH);
        return true;
    }

    bool write_char(char c) { return fd_ >= 0 && ::write(fd_, &c, 1) == 1; }

    // Whatever the Arduino has sent since the last call; empty if nothing is waiting.
    std::string read_available() {
        std::string out;
        if (fd_ < 0) return out;
        char buffer[256];
        for (;;) {
            const ssize_t n = ::read(fd_, buffer, sizeof(buffer));
            if (n <= 0) break;
            out.append(buffer, static_cast<size_t>(n));
            if (static_cast<size_t>(n) < sizeof(buffer)) break;
        }
        return out;
    }

    bool is_open() const { return fd_ >= 0; }

    void close() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    bool fail(std::string* error, const std::string& message) {
        if (error) *error = message;
        close();
        return false;
    }

    int fd_ = -1;
};

}  // namespace edge_tracking::apps
