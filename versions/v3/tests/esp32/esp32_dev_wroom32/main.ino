// ESP32 test sketch (Arduino)
// - Reads newline-terminated CSV from Serial at 115200
// - Parses CSV from run_csv_logger (frame_idx, ts_wallclock_ms, ts_source_ms, source, detected, target_id, cx, cy, err_px, err_norm, err_deg, bbox_h, est_dist_m, avg_score, fps)
// - Sends back ACK lines: "ACK,OK,<track_id>,<err_norm>,<err_deg>,<ts_ms>" or "ACK,NO_TARGET,,,<ts_ms>"
//
// Build/flash examples:
//  - Arduino IDE: select board "ESP32 Dev Module" and upload
//  - Arduino CLI:
//      arduino-cli compile --fqbn esp32:esp32:esp32 tests/esp32/esp32_dev_wroom32
//      arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 tests/esp32/esp32_dev_wroom32
//  - PlatformIO (platformio.ini snippet):
//      [env:esp32dev]
//      platform = espressif32
//      board = esp32dev
//      framework = arduino

const uint32_t BAUD_RATE = 115200;
const int MAX_FIELDS = 20;
const char *NO_TARGET_PREFIX = "ACK,NO_TARGET,,,";  // empty track_id, err_norm, err_deg to align with ACK,OK,<track_id>,<err_norm>,<err_deg>,<ts_ms>

int splitCsv(const String &line, String *outFields, int maxFields) {
  int fieldIndex = 0;
  int start = 0;
  const int len = line.length();
  for (int i = 0; i <= len; i++) {
    if (i == len || line.charAt(i) == ',') {
      if (fieldIndex < maxFields) {
        outFields[fieldIndex] = line.substring(start, i);
        outFields[fieldIndex].trim();
      }
      fieldIndex++;
      start = i + 1;
    }
  }
  return fieldIndex;
}

void setup() {
  Serial.begin(BAUD_RATE);
  while (!Serial) {
    delay(10);
  }
  Serial.println("ESP32 CSV responder ready");
}

void loop() {
  if (!Serial.available()) {
    delay(10);
    return;
  }
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) {
    return;
  }

  String fields[MAX_FIELDS];
  int count = splitCsv(line, fields, MAX_FIELDS);
  if (count < 5) {
    Serial.println("ACK,ERR,BAD_CSV");
    return;
  }

  String detected = fields[4];
  String trackId = (count > 5) ? fields[5] : "";
  String errNorm = (count > 9) ? fields[9] : "";
  String errDeg = (count > 10) ? fields[10] : "";
  String tsMs = (count > 1) ? fields[1] : "";

  bool hasTarget = detected == "true" || detected == "1";
  if (hasTarget) {
    Serial.print("ACK,OK,");
    Serial.print(trackId);
    Serial.print(",");
    Serial.print(errNorm);
    Serial.print(",");
    Serial.print(errDeg);
    Serial.print(",");
    Serial.println(tsMs);
  } else {
    // Keep three empty fields to align with ACK,OK,<id>,<err_norm>,<err_deg>,<ts_ms>
    Serial.print(NO_TARGET_PREFIX);
    Serial.println(tsMs);
  }
}
