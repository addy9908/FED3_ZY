/* new version for Arduino Uno with dataloger (01/08/2024: for 5V input/output)
  Pavlovian conditioning with tone + Saccharine (4kHz) or Air Puff(8kHZ) or nothing(white noise)
  ITI = 60s: need to make this changable by csv file
  Trial number = 60: need to make this changable by csv file
  
  Author: Zengyou Ye
  zengyou.ye@nih.gov
  December, 2023
*/

//#include <SD.h> // Include the SD library
#include "Arduino.h"
#include "RTClib.h"
#include <SdFat.h>
#include <Wire.h>
#include <SPI.h>
// #include <ArduinoLowPower.h>

RTC_PCF8523 rtc;
SdFat SD;

//input
//#define  triggerIn_Button   2  // Pin for manual trigger with button
#define  triggerIn_Ext      A1     // Pin for Exttrigger input
//output
#define  triggerOut         3             // rise to trigger and fall to stop
#define  cardSelect         10
#define  BUZZER             4            // Pin for CS tone
#define  CS                 5            // Pin for CS tone output recording
#define  Water              6             // Pin for reward
#define  Air                7             // Pin for punish
#define  triggerIn_LED      13     // Pin for Trigger received LEDs
//#define  VBATPIN            A7      //pin 9

File logfile;
File ITIfile;
File trialfile;

char filename[] = "ZY_MMDDYY_000.CSV";  // Array for file name data logged to named in setup
// for session
//bool newData = false; //receive mouseID from Bonsai
bool triggerOn = false; // State to track Exttrigger or button press
bool ignoreSD = false;
unsigned int toneFrequency = 0; // Initialize toneFrequency
unsigned int trialCount = 0; // Counter to limit loop iterations

// for trial
unsigned long previousMillis = 0;
unsigned long interval = 0; // Initialize ITI interval to 0

//preset the trial number and inter-trial-interval
unsigned long ITI=60000; // Initialize average ITI to 60s
unsigned int trialNumber=60; // Initialize average ITI to 60s

// // for frequency choose
int choices[] = {4,4,8,8,2}; //4k=water 8k=air,2=none
int number_of_choices = sizeof(choices) / sizeof(choices[0]);
unsigned int choicesMade = 0;

void setup() {
  Serial.begin(9600); // Initialize serial communication
  //pinMode(triggerIn_Button, INPUT); // Set button pin as input with pull-up resistor
  pinMode(triggerIn_Ext, INPUT);
  
  pinMode(triggerOut, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  pinMode(CS, OUTPUT);
  pinMode(Water, OUTPUT);
  pinMode(Air, OUTPUT);
  pinMode(triggerIn_LED, OUTPUT);
  pinMode(cardSelect, OUTPUT);
  
  Wire.begin();
  rtc.begin();
  randomSeed(analogRead(A2)); //randomSeed(0) for fixed seed
  
  delay(1000); // Allow time for serial monitor or computer connection
}

void loop() {
  unsigned long currentMillis = millis();
  while (triggerOn == false){
    checkTrigger();
    if (triggerOn){
      initiateSession();
    }
  }

  if (triggerOn){
    if (trialCount < trialNumber && (currentMillis-previousMillis) > interval) { // Limit loop iterations to 60    
      trialCount++;
      toneFrequency = RANDD(); // select RandomNameFromRemaining without replacement
      logEvent("Trial_Start");
      runTest();
      
      interval = ITI + random(-1000, 1000); // Random ITI between 59 to 61 seconds
      previousMillis = currentMillis; 
    }
    else if (trialCount >= trialNumber) {
      logEvent("Session_End");
      resetSession();
    }
  }
}

void checkTrigger() {
  triggerOn = false;
  
  if (digitalRead(triggerIn_Ext) == HIGH) { // Check if Exttrigger signal or button press for trial start
    delay (1);
	  if (digitalRead(triggerIn_Ext) == HIGH) {
      digitalWrite(triggerIn_LED, HIGH); 
      delay (25);
		  digitalWrite(triggerIn_LED, LOW);
      triggerOn = true; // Set trigger state to true if trigger detected
      // Serial.print("TriggerOn: ");
      // Serial.println(triggerOn);
    }
  }
}

void runTest() {
  delay(10000); // Wait for 10 seconds
  if (toneFrequency == 4) {
    tone(BUZZER, toneFrequency * 1000); // Play tone at specified frequency
    digitalWrite(CS, HIGH);
    logEvent("CS_ON"); // Record data
    delay(1700); //c start after finish tone, overlap 0.3s
    digitalWrite(Water, HIGH);
    logEvent("Water_On"); // Record data
    delay(300); // 
    noTone(BUZZER); // Stop tone at 2s
    digitalWrite(CS, LOW);
    logEvent("CS_Off"); // Record data
    delay(200); // Adjust this delay for the trigger duration
    digitalWrite(Water, LOW);// Water for 0.5s
    logEvent("Water_Off"); // Record data
  }
  else if (toneFrequency == 8) {
    tone(BUZZER, toneFrequency * 1000); // Play tone at specified frequency
    digitalWrite(CS, HIGH);
    logEvent("CS_ON"); // Record data
    delay(1700); //c start after finish tone, overlap 0.3s
    digitalWrite(Air, HIGH);
    logEvent("Air_On"); // Record data
    delay(300); // 
    noTone(BUZZER); // Stop tone at 2s
    digitalWrite(CS, LOW);
    logEvent("CS_Off"); // Record data
    delay(200); // Adjust this delay for the trigger duration
    digitalWrite(Air, LOW);// Water for 0.5s
    logEvent("Air_Off"); // Record data
  }
  else{
    digitalWrite(CS, HIGH);
    logEvent("CS_ON"); // Record data
    Noise(2000);
    digitalWrite(CS, LOW); 
    logEvent("CS_Off");
  }
}

int RANDD(){ //choose without replacement
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

void initiateSession() {                        // previous session infor will be on before next
  //newData = false; 
  CreateDataFile();
  // Serial.println("Request_mouseID");
  // if (newData = false){
  //   Serial.println("Request_mouseID");
  //   recvWithStartEndMarkers();
  // }
  //Serial.println(filename);

  Serial.println("Start_Recording");
  writeHeader();
  toneFrequency = 0; // Initialize toneFrequency
  trialCount = 0; // Counter to limit loop iterations

  // for trial
  //CS = 0;
  previousMillis = 0;
  interval = 0; // Initialize ITI interval to 0

  // for frequency choose
  choicesMade = 0;
  logEvent("Session_Start");
  
  digitalWrite(triggerOut, HIGH); //on for start and off for end
  logEvent("Trigger_Out");
}

void resetSession(){
  digitalWrite(triggerOut, LOW);
  triggerOn = false;
  ignoreSD = false;
  //Serial.println("End_Recording");
}

void Noise(int duration) {
  // White noise to signal errors
  for (int i = 0; i < duration/50; i++) {
    tone (BUZZER, random(50, 250));
    delay(50);
  }
  noTone(BUZZER);
}

// //  dateTime function
// void dateTime(uint16_t* date, uint16_t* time) {
//   DateTime now = rtc.now();
//   // return date using FAT_DATE macro to format fields
//   *date = FAT_DATE(now.year(), now.month(), now.day());

//   // return time using FAT_TIME macro to format fields
//   *time = FAT_TIME(now.hour(), now.minute(), now.second());
// }

void getFilename(char *filename) {
  DateTime now = rtc.now();

  filename[3] = now.month() / 10 + '0';
  filename[4] = now.month() % 10 + '0';
  filename[5] = now.day() / 10 + '0';
  filename[6] = now.day() % 10 + '0';
  filename[7] = (now.year() - 2000) / 10 + '0';
  filename[8] = (now.year() - 2000) % 10 + '0';
  filename[13] = '.';
  filename[14] = 'C';
  filename[15] = 'S';
  filename[16] = 'V';
  for (uint8_t i = 0; i < 100; i++) {
    filename[10] = '0' + i / 100;
    filename[11] = '0' + (i%100) / 10;
    filename[12] = '0' + i % 10;

    if (! SD.exists(filename)) {
      break;
    }
  }
  return;
}

void CreateDataFile () {
  // Initialize SD card and create the datafile
  //SdFile::dateTimeCallback(dateTime);
  if (!SD.begin(cardSelect)) {
    Serial.println("Card failed, or not present");
    ignoreSD=true;
    ITI=60000;
    trialNumber = 60;  
  }
  else {
    // create files if they dont exist and grab ITI and trialNUmber
    Serial.println("SD card initialized.");
    //Serial.println(DisplayDateTime());
    
    // ITIfile = SD.open("InterTrialInterval.csv", FILE_WRITE);
    // ITIfile = SD.open("InterTrialInterval.csv", FILE_READ);
    // ITI = ITIfile.parseInt();
    // ITIfile.close();
    // if (ITI==0){
    //   ITI=60000; //default
    // }
    

    // trialfile = SD.open("trialNumber.csv", FILE_WRITE);
    // trialfile = SD.open("trialNumber.csv", FILE_READ);
    // trialNumber = trialfile.parseInt();
    // trialfile.close();
    // if (trialNumber == 0){
    //   trialNumber = 60; //default
    // }
  
    // Name filename in format F###_MMDDYYNN, where MM is month, DD is day, YY is year, and NNN is an incrementing number for the number of files initialized each day
    //strcpy(filename, "ZY_MMDDYY_NNN.CSV");  // placeholder filename ZY_F01_S1_MMDDYY_NNN.CSV
    getFilename(filename);
  }
}

String DisplayDateTime(){
  // Print date and time at bottom of the screen
  DateTime now = rtc.now();
  String timeString = String(now.month()) + "/" + String(now.day()) + "/" + String(now.year()) + " ";

  if (now.hour() < 10)
    timeString = timeString + '0';      // Trick to add leading zero for formatting
  timeString = timeString + String(now.hour()) + ":";
  if (now.minute() < 10)
    timeString = timeString + '0';      // Trick to add leading zero for formatting
  timeString = timeString + String(now.minute()) + ":";
  if (now.second() < 10)
    timeString = timeString + '0';
  timeString = timeString + String(now.second());
  return timeString;
}

void writeHeader() {  
  if (!ignoreSD){
    logfile = SD.open(filename, FILE_WRITE);
    if ( ! logfile ) {
      Serial.println("Error_opening_logfile");
    }
    else{
      Serial.print("Sucess_ opening_logfile: ");
      Serial.println(filename);
      logfile.println("TimeStamp, millis, Trial, Event, ToneFrequency, Tone, Water, Air, TriggerIn");
      logfile.close();
    }
  }else{
    Serial.println("Ignore_SD");
  }
  Serial.println("TimeStamp,millis, Trial, Event, ToneFrequency, Tone, Water, Air, TriggerIn");
}

void logEvent(String Event) {
  String timestamp = DisplayDateTime();
  unsigned long millisec = millis(); // Get timestamp (in milliseconds)
  //Series
  Serial.print(timestamp);
  Serial.print(",");
  Serial.print(millisec);
  Serial.print(",");
  Serial.print(trialCount);
  Serial.print(",");
  Serial.print(Event);
  Serial.print(",");
  Serial.print(toneFrequency);
  Serial.print(",");
  Serial.print(digitalRead(CS));
  Serial.print(",");
  Serial.print(digitalRead(Water));
  Serial.print(",");
  Serial.print(digitalRead(Air));
  Serial.print(",");
  Serial.println(digitalRead(triggerIn_Ext));

  if (!ignoreSD){
    //fix filename (the .CSV extension can become corrupted) and open file
    filename[13] = '.';
    filename[14] = 'C';
    filename[15] = 'S';
    filename[16] = 'V';

    //SD
    logfile = SD.open(filename, FILE_WRITE);
    if(logfile){
      logfile.print(timestamp);  
      logfile.print(",");
      logfile.print(millisec);
      logfile.print(",");
      logfile.print(trialCount);
      logfile.print(",");
      logfile.print(Event);
      logfile.print(",");
      logfile.print(toneFrequency);
      logfile.print(",");
      logfile.print(digitalRead(CS));
      logfile.print(",");
      logfile.print(digitalRead(Water));
      logfile.print(",");
      logfile.print(digitalRead(Air));
      logfile.print(",");
      logfile.println(digitalRead(triggerIn_Ext));

      logfile.flush();
      logfile.close();
    }
  }
}