int redPin = 2;
int yellowPin = 3;
int greenPin = 4;

void setup() {
  pinMode(redPin, OUTPUT);
  pinMode(yellowPin, OUTPUT);
  pinMode(greenPin, OUTPUT);

  Serial.begin(9600);
  setLights('R'); 
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'R' || cmd == 'G' || cmd == 'Y' || cmd == 'N') {
      setLights(cmd);
    }
  }
}

void setLights(char mode) {
  digitalWrite(redPin, LOW);
  digitalWrite(yellowPin, LOW);
  digitalWrite(greenPin, LOW);

  if (mode == 'R') {
    digitalWrite(redPin, HIGH);
  } 
  else if (mode == 'G') {
    digitalWrite(greenPin, HIGH);
  } 
  else if (mode == 'Y') {
    unsigned long start = millis();
    while (millis() - start < 2000) {
      digitalWrite(yellowPin, HIGH);
      delay(300);
      digitalWrite(yellowPin, LOW);
      delay(300);
    }
    digitalWrite(redPin, HIGH);
  }
}

