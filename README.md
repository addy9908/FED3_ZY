# 1. FED3_ZY: [TimedFreeFeed_ZY_20231218.ino](Firmware/TimedFreeFeed_ZY_20231218/TimedFreeFeed_ZY_20231218.ino)
# A Timed Free_Feeding Lab Project Created by Zengyou Ye @ NIH/NIDA/IRP

This lab project enables free feeding for a specified duration each day upon receiving a TTL trigger. The system then waits for another trigger after the session ends.

## Trigger Handling
- A BNC connector listens for incoming triggers.
- When `BNCinput = true`, it switches back to sending a signal output.

## Session Configuration
- Users can set the `sessionDuration` at the start, similar to configuring `FEDID`.  
- By default, the session duration is set to **3 hours (10,800 seconds)**, as displayed on the screen.  
- The screen also shows the trigger's start time in **hour:minute** format.

## Output Behavior
When the BNC connector is in output mode:  
- A **200 ms TTL signal** is sent each time a pellet is taken.  
- A **100 ms TTL signal** is sent upon receiving a trigger.

## User Instructions
1. Replace the original `FED3.cpp` and `FED3.h` files with those in the [**Firmware\Modified_Source_Code\20231218_numMotorTurns**](Firmware/Modified_Source_Code/20231218_numMotorTurns) folder.  
2. Upload the [**TimedFreeFeed_ZY_20231218.ino**](Firmware/TimedFreeFeed_ZY_20231218/TimedFreeFeed_ZY_20231218.ino) file to the FED3 device.  
3. If needed, adjust the `sessionDuration` from its default of 3 hours.

# 2. Optional: [Timed_FF_ZY.ino](Firmware/Timed_FF_ZY/Timed_FF_ZY.ino): 
This special edition is designed to control the FED3 using its internal timer, allowing users to set the `start time` and `session duration`.

