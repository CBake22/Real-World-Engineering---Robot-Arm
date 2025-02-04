const int fanPin = 9;  // Digital pin connected to transistor base via a resistor

void setup() {
  pinMode(fanPin, OUTPUT);  // Set pin as an output
  digitalWrite(fanPin, LOW); // Ensure fan is off initially
}

void loop() {
  digitalWrite(fanPin, HIGH); // Turn the fan on (transistor conducts)
  delay(5000);                // Fan runs for 5 seconds

  digitalWrite(fanPin, LOW);  // Turn the fan off (transistor stops conducting)
  delay(5000);                // Fan stays off for 5 seconds
}
