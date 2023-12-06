//#include <SD.h> // Include the SD library
//input
const int manualTriggerIn = 2;  // Pin for manual trigger with button
const int extTriggerIn = 3;     // Pin for Exttrigger input

//output
const int triggerOut = 4;             // Pin for CS tone
const int csOut = 5;             // Pin for CS tone
const int usOut1 = 6;             // Pin for reward
const int usOut2 = 7;             // Pin for punish
const int triggerInLED = 8;     // Pin for Trigger received LED
//const int chipSelect = 10;      // Pin used for SD card module's chip select

//File dataFile;
// for session
bool triggerOn = false; // State to track Exttrigger or button press
unsigned int toneFrequency = 0; // Initialize toneFrequency
unsigned int loopCount = 0; // Counter to limit loop iterations

// for trial
unsigned long previousMillis = 0;
unsigned long interval = 0; // Initialize ITI interval to 0
//preset the trial number and inter-trial-interval
unsigned int ITI = 60000; // Initialize average ITI to 60s
unsigned int trialNumber = 60; // Initialize average ITI to 60s

// for frequency choose
int choices[] = {4,4,4,4,4,8,8,8,8,8};
int number_of_choices = sizeof(choices) / sizeof(choices[0]);
int choicesMade = 0;

void setup() {
  pinMode(manualTriggerIn, INPUT_PULLUP); // Set button pin as input with pull-up resistor
  pinMode(extTriggerIn, INPUT_PULLUP);
  
  pinMode(triggerOut, OUTPUT);
  pinMode(csOut, OUTPUT);
  pinMode(usOut1, OUTPUT);
  pinMode(usOut2, OUTPUT);
  pinMode(triggerInLED, OUTPUT);
  
  randomSeed(analogRead(A0)); //randomSeed(0) for fixed seed
  Serial.begin(9600); // Initialize serial communication
  delay(1000); // Allow time for serial monitor or computer connection

  // if (!SD.begin(chipSelect)) {
  //   Serial.println("SD card initialization failed!");
  //   return;
  // }

  // dataFile = SD.open("data.csv", FILE_WRITE); // Open the file for writing

  // if (dataFile) {
  //   dataFile.println("Times(ms),usOut1,usOut2,Tone,ToneFrequency,ExtTrigger"); // Write header
  //   dataFile.close(); // Close the file
  // } else {
  //   Serial.println("Error opening file!");
  // }
}

void loop() {
  unsigned long currentMillis = millis();

  if (loopCount < trialNumber && (currentMillis-previousMillis > interval)) { // Limit loop iterations to 60
    if (!triggerOn){
      checkTrigger();
    }
    if (triggerOn){
      Serial.print(">>>Start Trial Number: ");
      Serial.println(loopCount+1);          
      runTest();
      loopCount++;
      interval = ITI + random(-1000, 1000); // Random ITI between 59 to 61 seconds
      previousMillis = currentMillis; 
    }
  }else if (loopCount >= trialNumber) {
    Serial.print("===Session ends===");
    triggerOn = false;
    interval = 0;
    loopCount = 0;
  }
}

void checkTrigger() {
  if (digitalRead(extTriggerIn) == HIGH || digitalRead(manualTriggerIn) == HIGH) { // Check if Exttrigger signal or button press for trial start
    delay(200); // Additional delay to ensure the signal is properly captured
    digitalWrite(triggerInLED, HIGH); // Light up LED 1 as long as the duration of the trigger
    triggerOn = true; // Set trigger state to true if trigger detected
    Serial.print("===Session starts===");
    logEvent(); // Record data
  }
  digitalWrite(triggerInLED, LOW); // Light up LED 1 as long as the duration of the trigger
}

void runTest() {
  delay(10000); // Wait for 10 seconds

  int toneFrequency = selectRandomNameFromRemaining(); // Randomly select tone frequency (8kHz or 12kHz)
  tone(csOut, toneFrequency * 1000); // Play tone at specified frequency
  logEvent(); // Record data
  
  delay(1700); //c start after finish tone, overlap 0.3s
  if (toneFrequency == 4) {
    digitalWrite(usOut1, HIGH);
    logEvent(); // Record data
    delay(300); // 
    noTone(csOut); // Stop tone at 2s
    logEvent(); // Record data
    delay(200); // Adjust this delay for the trigger duration
    digitalWrite(usOut1, LOW);// usOut1 for 0.5s
    logEvent(); // Record data
  }else if (toneFrequency == 8) {
    digitalWrite(usOut2, HIGH);
    logEvent(); // Record data
    delay(300); // 
    noTone(csOut); // Stop tone at 2s
    logEvent(); // Record data
    delay(200); // Adjust this delay for the trigger duration
    digitalWrite(usOut2, LOW);// usOut1 for 0.5s
    logEvent(); // Record data
  } 
}

int selectRandomNameFromRemaining(){
  if(choicesMade >= number_of_choices)
  {
    choicesMade = 0; //after finish one round, start over
  }
  int selection = random(choicesMade, number_of_choices);
  int temp = choices[choicesMade];
  choices[choicesMade] = choices[selection];
  choices[selection] = temp;
  return choices[choicesMade++]; //moved incrementing choicesMade to here
}

void logEvent() {
  unsigned long timestamp = millis(); // Get timestamp (in milliseconds)
  int triggerOutState = digitalRead(triggerOut);
  int usOut1State = digitalRead(usOut1);
  int usOut2State = digitalRead(usOut2);
  int toneState = digitalRead(csOut);
  int externaltriggerOn = digitalRead(extTriggerIn); //could use for framerate of imaging

  Serial.print("Times (ms): ");
  Serial.print(timestamp);
  Serial.print(" | triggerOut: ");
  Serial.print(triggerOutState);
  Serial.print(" | usOut1: ");
  Serial.print(usOut1State);
  Serial.print(" | usOut2: ");
  Serial.print(usOut2State);
  Serial.print(" | Tone: ");
  Serial.print(toneState);
  Serial.print(" | ToneFrequency: ");
  Serial.print(toneFrequency);
  Serial.print(" | ExtTrigger: ");
  Serial.println(externaltriggerOn);

  // dataFile = SD.open("data.csv", FILE_WRITE);
  // if (dataFile) {
  //   dataFile.print(timestamp);
  //   dataFile.print(",");
  //   dataFile.print(usOut1State);
  //   dataFile.print(",");
  //   dataFile.print(usOut2State);
  //   dataFile.print(",");
  //   dataFile.print(toneState);
  //   dataFile.print(",");
  //   dataFile.print(toneFrequency);
  //   dataFile.print(",");
  //   dataFile.println(externaltriggerOn);
  //   dataFile.close();
  // } else {
  //   Serial.println("Error opening file!");
  // }
}
