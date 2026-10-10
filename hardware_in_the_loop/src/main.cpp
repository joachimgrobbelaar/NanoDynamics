#include <Arduino.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "transition_model_data.h"

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 32 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

void setup() {
  Serial.begin(115200);
  while (!Serial);

  model = tflite::GetModel(transition_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided is schema version not equal to supported version!");
    while (1);
  }

  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddFullyConnected();
  resolver.AddTanh();
  resolver.AddAdd();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("READY");
}

void loop() {
  if (Serial.available() >= sizeof(float) * 6) {
    float state_in[6];
    Serial.readBytes((char*)state_in, sizeof(state_in));

    for (int i = 0; i < 6; i++) {
      input->data.f[i] = state_in[i];
    }

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Invoke failed");
      return;
    }

    float state_out[6];
    for (int i = 0; i < 6; i++) {
      state_out[i] = output->data.f[i];
    }

    Serial.write((uint8_t*)state_out, sizeof(state_out));
  }
}
