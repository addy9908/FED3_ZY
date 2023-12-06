# FED3_ZY
This is for Lab project to enable a free feeding for a certain time during a day when receive a TTL trigger, then wait for another trigger after session ends:

1. BNC is used for listening to a trigger, and switch back to send signal out when BNCinput=true
2. User can set the sessionDuration at the begining as how they setup FEDID, the default is 3hr as showed on the screen (10800s)
3. The screen will also show the trigger start time as hour:minute.
4. When BNC_OUT as output, it will send TTL out anytime when the pellet is taken (200ms), and when the trigger received (100ms).

Credit to Zengyou Ye @ NIH/NIDA/IRP
