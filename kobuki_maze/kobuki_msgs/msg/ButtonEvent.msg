# Provides a button event.
# This message is generated whenever a particular button is pressed or released.
# Note that, despite buttons field on SensorState messages, state field is not a
# bitmask, but the new state of a single button.

uint8 Button0 = 0
uint8 Button1 = 1
uint8 Button2 = 2

uint8 RELEASED = 0
uint8 PRESSED  = 1

uint8 button
uint8 state
