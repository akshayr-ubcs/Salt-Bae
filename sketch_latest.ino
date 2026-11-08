#include <Arduino_RouterBridge.h>

// Hardware Pin Definitions
const int TRIG_PIN = 8;
const int ECHO_PIN = 7;
const int SERVO_PIN = 9;

// Baseline distance from HC-SR04 face to empty deck surface (mm)
const float DECK_BASELINE_MM = 150.0;

// Readings below this are treated as sensor noise, not ice (mm)
const float NOISE_FLOOR_MM = 3.0;

// Safety cap on how long the gate can stay open
const int MAX_OPEN_MS = 5000;

// Raw distance (mm) from the most recent measurement, -1 if no valid reading.
// get_ice_thickness() updates it; get_last_distance_mm() reports it.
float lastDistanceMm = -1.0;

// Replacement for pulseIn(): measures how long 'pin' stays at 'level', in microseconds.
// Returns 0 if nothing happens within timeoutUs.
unsigned long pulseInUs(int pin, int level, unsigned long timeoutUs) {
  unsigned long start = micros();

  // wait for any pulse already in progress to end
  while (digitalRead(pin) == level) {
    if (micros() - start > timeoutUs) return 0;
  }
  // wait for the pulse to start
  while (digitalRead(pin) != level) {
    if (micros() - start > timeoutUs) return 0;
  }
  // measure how long it stays at that level
  unsigned long pulseStart = micros();
  while (digitalRead(pin) == level) {
    if (micros() - pulseStart > timeoutUs) return 0;
  }
  return micros() - pulseStart;
}

// Send servo pulses at 50 Hz for 'ms' milliseconds.
// Keeping the pulses going is what makes the servo hold its position.
// 0 deg = ~1000us pulse (sealed gate), 90 deg = ~1500us pulse (open chute)
void holdServoAngle(int angle, unsigned long ms) {
  if (angle < 0) angle = 0;
  if (angle > 180) angle = 180;
  int pulseWidthUs = 1000 + (angle * 500 / 90);

  unsigned long end = millis() + ms;
  while ((long)(millis() - end) < 0) {
    digitalWrite(SERVO_PIN, HIGH);
    delayMicroseconds(pulseWidthUs);
    digitalWrite(SERVO_PIN, LOW);
    delayMicroseconds(20000 - pulseWidthUs);
  }
}

// One raw distance reading in mm, or -1 if no echo
float readDistanceMm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long durationUs = pulseInUs(ECHO_PIN, HIGH, 30000);
  if (durationUs == 0) return -1.0;
  return (durationUs * 0.343) / 2.0;
}

// Ice thickness in mm, using the median of 5 readings to ignore outliers.
// Also stores the median raw distance in lastDistanceMm.
float get_ice_thickness() {
  float r[5];
  int n = 0;
  for (int i = 0; i < 5; i++) {
    float d = readDistanceMm();
    if (d > 0) r[n++] = d;
    delay(15);
  }
  if (n < 3) {          // too few good readings to trust
    lastDistanceMm = -1.0;
    return 0.0;
  }

  // simple sort, then take the middle value
  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      if (r[j] < r[i]) {
        float t = r[i];
        r[i] = r[j];
        r[j] = t;
      }
    }
  }
  float distanceMm = r[n / 2];
  lastDistanceMm = distanceMm;

  float thickness = DECK_BASELINE_MM - distanceMm;
  return (thickness > NOISE_FLOOR_MM) ? thickness : 0.0;
}

// Raw distance (mm) behind the last get_ice_thickness() call, -1 if no echo.
// Does not take a new measurement, so the two numbers always match.
float get_last_distance_mm() {
  return lastDistanceMm;
}

// Gate control triggered via Bridge RPC from Python
int dispense_salt(int duration_ms) {
  if (duration_ms <= 0) return 0;
  if (duration_ms > MAX_OPEN_MS) duration_ms = MAX_OPEN_MS;

  holdServoAngle(90, duration_ms);  // open, holding position the whole time
  holdServoAngle(0, 800);           // close and give the horn time to fully settle
  return 1;
}

void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(SERVO_PIN, OUTPUT);

  holdServoAngle(0, 400);

  Bridge.begin();
  Bridge.provide("get_ice_thickness", get_ice_thickness);
  Bridge.provide("get_last_distance_mm", get_last_distance_mm);
  Bridge.provide("dispense_salt", dispense_salt);
}

void loop() {
  Bridge.update();
  // Keep sending "closed" pulses between doses (one 20 ms servo frame per pass),
  // so the gate is held shut instead of going limp when no signal is sent.
  holdServoAngle(0, 20);
}
