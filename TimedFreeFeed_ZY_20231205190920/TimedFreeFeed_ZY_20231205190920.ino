/*
  Feeding experimentation device 3 (FED3)
  timed free feeding with external trigger by Zengyou Ye

  zengyou.ye@nih.gov
  December, 2023
*/

#include <FED3.h>                               //Include the FED3 library 

String sketch = "Timed_FF";               //Unique identifier text for each sketch (this will show up on the screen and in log file)
FED3 fed3 (sketch);                             //Start the FED3 object
//int pelletLimit[]={};                         // for future pellet limit exp
//int sessionCount = 0;                         // for future pellet limit exp

//unsigned long sessionDuration = 180000;           // session length 3min

void setup() {
  fed3.DisplayPokes = false;
  fed3.disableSleep(); 
  //  fed3.sessionDuration = sessionDuration;
  fed3.setDuration = true;  
  fed3.begin();                 
}

void loop() {
  fed3.run();
  if (fed3.BNCinput == false){ 
    displayMessage("Wait",65);
    displayMessage("Trig",85);
    displayMessage(String(fed3.sessionDuration/1000),105);
    fed3.ReadBNC(false);                        //set BNCinput=true, don't blinkgreen
    if (fed3.BNCinput){
      initiateSession(100);
      displayMessage("Start:",65);
      displayMessage(timeString(),85);      
    }
  }

  if (fed3.BNCinput == true){                   //Creates timer that starts when BNC input is recieved and updates following BNC input
    if (fed3.sessionTimer<fed3.sessionDuration){                   //Keeps house lights on for duration of session (60 minutes)
    //add here for future to check if fed3.PelletCount>=pelletLimit[sessionCount]
      fed3.Feed(200);                           //Deliver pellet,will send a 200ms pulse when the pellet is taken.
      //fed3.Timeout(5);                          //5s timeout  
      //displayMessage(String(fed3.sessionTimer/1000),105);            
    }
    else{
      displayMessage("End!",105);
      resetSession();
    }
  }
  //controlSleep();   //For free-feeding task, only sleep when pellet is in the well or sessionstart=false
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
  fed3.BNCinput=false;
  fed3.enableSleep(); 
  fed3.goToSleep();  //may directly use fed3.ReleaseMotor() to save battery
}

void initiateSession(int pulse) {                        // previous session infor will be on before next
  //sessionCount++;                         //next pelletlimit will be: pelletLimit[sessionCount]
  pinMode(BNC_OUT, OUTPUT);                 //switch back for output
  fed3.disableSleep(); 
  if (pulse>0){
    fed3.BNC(BNC_OUT,1);
  }
  fed3.sessionTimer = 0;                    //or millis()-fed3.sessionStartTime
  fed3.Tone(4000,2000) ;                    //play 4kHz tone for 5s, may move to the end of if
  fed3.Event="Start";
  fed3.PelletCount = 0;
  fed3.LeftCount = 0;
  fed3.RightCount = 0;
  fed3.BlockPelletCount = 0;
  fed3.logdata();
  fed3.UpdateDisplay();
}

void controlSleep(){
  if (digitalRead(1) == LOW || !fed3.BNCinput) {  // Check if pellet-well is empty (pellet-well beambreak reads HIGH when empty)
    fed3.enableSleep();    // If so,enable sleep
  } else {                       // But if there's a pellet... 
    fed3.disableSleep();     // disable sleep
  }
}

