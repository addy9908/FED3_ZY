/*
  Free feeding without trigger, but with internal timer_Zengyou Ye

  Feeding experimentation device 3 (FED3)
  Fixed Ratio 1 with a user-settable Timed Feeding option
  Device starts in FR1
  Hold both pokes at startup to enter a small edit menu where you can change the device number and the timed-feeding hours
  Device will only dispense pellets during these hours, although it will log pokes at all times
 
  alexxai@wustl.edu
  October, 2021

  This project is released under the terms of the Creative Commons - Attribution - ShareAlike 3.0 license:
  human readable: https://creativecommons.org/licenses/by-sa/3.0/
  legal wording: https://creativecommons.org/licenses/by-sa/3.0/legalcode
  Copyright (c) 2020 Lex Kravitz
*/

#include <FED3.h>                                                                      //Include the FED3 library 
String sketch = "FF_timedIS_ZY";                                                           //Unique identifier text for each sketch
FED3 fed3 (sketch);                                                                    //Start the FED3 object
bool session_start = false;

void setup() {
  fed3.setTimed = true;                                                                //Set a flag to ask FED3 to edit "Timed Feeding" times when it enters the "Set Device Number" edit menu (this defaults to false)
  fed3.setDuration = true;  
  
  fed3.DisplayPokes = false;                            //Customize the DisplayPokes option to 'false' to not display the poke indicators
  fed3.disableSleep(); 
  fed3.begin();                                                                        //Setup the FED3 hardware
}

void loop() {
  fed3.run();                                                                          //Call fed.run at least once per loop
  if (session_start == false){
    displayMessage("Stop",65);
    String sessionDuration = String(fed3.sessionDuration/3600000)+ ":" + String((fed3.sessionDuration%3600000)/60000);
    displayMessage(sessionDuration,85); //min
    
    if (fed3.currentHour >= fed3.timedStart){
      initiateSession(100);
      displayMessage("Start:",65);
      displayMessage(timeString(),85); 
    }
  }

  if (session_start == true){
    if (fed3.sessionTimer<fed3.sessionDuration && fed3.numMotorTurns<40){                   //Keeps house lights on for duration of session (60 minutes), jamclear 3 times
    //add here for future to check if fed3.PelletCount>=pelletLimit[sessionCount]
      fed3.Feed(200);                           //Deliver pellet,will send a 200ms pulse when the pellet is taken.
      //fed3.Timeout(5);                          //5s timeout  
      //displayMessage(String(fed3.sessionTimer/1000),105);            
    }
    else{
      displayMessage("End!",65);
      displayMessage(timeString(),85);
      resetSession();
    }
  }                                                               //Deliver pellet
}

String timeString() {
  DateTime now = fed3.now();
  String timeString = String(now.hour()) + ":" + String(now.minute());
  return timeString;
}

void displayMessage(String message, int line) {
  fed3.display.fillRect (115, line-15, 42, 20, WHITE); // line 20,erase the data on screen without clearing the entire screen by pasting a white box over it
  fed3.display.setCursor(115, line);    
  fed3.display.print(message);
  //fed3.display.setTextSize(1);
  //fed3.display.setFont(&Org_01);  
  fed3.display.refresh();
}

void resetSession() {
  session_start = false;
}

void initiateSession(int pulse) {                        // previous session infor will be on before next
  session_start == true;
  fed3.sessionStartTime = millis();
  pinMode(BNC_OUT, OUTPUT);                 //switch back for output
  fed3.disableSleep(); 
  if (pulse>0){
    fed3.BNC(pulse,1);
  }
  fed3.sessionTimer = 0;                    //or millis()-fed3.sessionStartTime
  //fed3.Tone(4000,10000) ;                    //play 4kHz tone for 10s, may move to the end of if
  fed3.Event="Start";
  fed3.numMotorTurns = 0;
  fed3.PelletCount = 0;
  //fed3.jamClearTime = 0;
  //fed3.lastPellet = fed3.now().unixtime();
  fed3.LeftCount = 0;
  fed3.RightCount = 0;
  fed3.BlockPelletCount = 0;
  fed3.logdata();
  fed3.UpdateDisplay();

  // light and sound
  digitalWrite(GREEN_LED, HIGH);
  fed3.pixelsOn(0,0,0,10); //zy: W (RGBW)
  tone (BUZZER, 4000,10000); //zy
  fed3.pixelsOn(0,0,0,0); //zy
  digitalWrite(GREEN_LED, LOW);
}