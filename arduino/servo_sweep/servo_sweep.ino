// servo_sweep.ino - SG90 sweep driven by the Jetson's person detector.
//
// Wiring (Arduino Uno/Nano):
//   SG90 orange (signal) -> D9        D9/D10 are the Servo library's Timer1 pins
//   SG90 red    (+5V)    -> 5V        one small servo, unloaded, is fine on USB power;
//                                     add a separate 5V supply if it stutters or resets the board
//   SG90 brown  (GND)    -> GND       tie the supply ground to the Arduino ground
//
// Protocol (115200 baud, one character per command, upper or lower case):
//   'S' -> sweep while the sender keeps talking; stops on its own after SERIAL_TIMEOUT_MS
//          of silence, so the servo cannot run away if the Jetson app dies. This is what
//          servo_track sends, and it repeats it every 500 ms.
//   'L' -> latched sweep: keeps going with no further commands. For testing by hand from
//          the Serial Monitor, where you press Send once.
//   'X' -> stop, centre, and release the servo.
//   'P' -> ping, replies "ok".

#include <Servo.h>

const int SERVO_PIN = 9;
const int ANGLE_MIN = 30;  // sweep limits in degrees; an SG90 is happiest well inside 0-180
const int ANGLE_MAX = 150;
const int ANGLE_CENTRE = 90;
const int STEP_DEG = 2;
const unsigned long STEP_MS = 15;             // 2 deg every 15 ms -> about 0.9 s per pass
const unsigned long SERIAL_TIMEOUT_MS = 3000;
const unsigned long DETACH_AFTER_MS = 1000;   // stop driving once centred, so the SG90 stops buzzing

Servo servo;
bool sweeping = false;
bool latched = false;  // sweeping without a watchdog, after 'L'
int angle = ANGLE_CENTRE;
int step = STEP_DEG;
unsigned long last_step_ms = 0;
unsigned long last_command_ms = 0;
unsigned long idle_since_ms = 0;
bool attached = false;

void attachIfNeeded() {
  if (!attached) {
    servo.attach(SERVO_PIN);
    attached = true;
  }
}

void setSweeping(bool on, bool latch) {
  latched = on && latch;
  if (on == sweeping) return;
  sweeping = on;
  attachIfNeeded();
  if (sweeping) {
    Serial.println(latched ? "sweep (latched)" : "sweep");
  } else {
    angle = ANGLE_CENTRE;
    servo.write(ANGLE_CENTRE);
    idle_since_ms = millis();
    Serial.println("idle");
  }
}

void setup() {
  Serial.begin(115200);
  attachIfNeeded();
  servo.write(ANGLE_CENTRE);
  last_command_ms = millis();
  idle_since_ms = millis();
  Serial.println("ready");
}

void loop() {
  while (Serial.available() > 0) {
    const char c = Serial.read();
    if (c == 'S' || c == 's') {
      last_command_ms = millis();
      setSweeping(true, false);
    } else if (c == 'L' || c == 'l') {
      last_command_ms = millis();
      setSweeping(true, true);
    } else if (c == 'X' || c == 'x') {
      last_command_ms = millis();
      setSweeping(false, false);
    } else if (c == 'P' || c == 'p') {
      last_command_ms = millis();
      Serial.println("ok");
    }
  }

  const unsigned long now = millis();
  if (sweeping && !latched && now - last_command_ms > SERIAL_TIMEOUT_MS) {
    Serial.println("timeout");
    setSweeping(false, false);
  }

  if (sweeping) {
    if (now - last_step_ms >= STEP_MS) {
      last_step_ms = now;
      angle += step;
      if (angle >= ANGLE_MAX) {
        angle = ANGLE_MAX;
        step = -STEP_DEG;
      } else if (angle <= ANGLE_MIN) {
        angle = ANGLE_MIN;
        step = STEP_DEG;
      }
      servo.write(angle);
    }
  } else if (attached && now - idle_since_ms > DETACH_AFTER_MS) {
    servo.detach();  // quiet while idle, and the horn turns freely by hand
    attached = false;
  }
}
