# PRPL TidyBot Onboarding
Written by Josh Roy

# Goals

After reading this document, a user should be able to:

1. Safely turn on and off the tidybot in the prpl lab space
2. Safely charge batteries, so that they can repeat experiments independently
3. Connect to the PRPL Tidybot and run pre-existing teleop + pre-programmed scripts

# Non-Goals

1. Write any bespoke code themselves
2. Be able to re-create the tidybot setup

# (Robot + System) Architecture Overview

## Physical Devices + Networking

As Princeton maintains a very strict, high quality (read: annoying for hacking on) network infrastructure, the PRPL lab has it's own router + network. Ask a PhD student or Post-doc to help you connect to it.

Within the PRPL lab, there are the following devices:

1. Tidybot (`tidybot@tidybot-nuc-prpl`)
2. Tidybot laptop (`yixuan@tidybot-laptop-prpl`)
3. Your laptop you are working on (`<username>@<laptopname>`)

If any of these three are down (ex: disconnected from network, powered off, etc, the system will not work)

## Tidybot Physical Architecture

PRPL's TidyBot is a First Robotics Challenge (FRC) Base with a Kinova Arm on top.

### Base

1. Camping Battery (to power the onboard computer + arm)
2. FRC Car Battery (to power the base motors)
3. NUC (small onboard computer)

### Arm

Kinova arm, nothing non-standard

# Power Up Process

### Base

1. Ensure everything is unplugged from their corresponding chargers/wall outlets
2. Ensure the robot is in a safe place where sudden movements will not harm people or objects
3. Plug in the car battery (TODO: Josh takes + inserts photo).
4. Flip the switch to enable the car battery (TODO: Josh insert photo) The lights on the main board plugged into the wheels should light up (TODO: Josh inserts photo)
4. Turn on the camping battery (TODO: Josh inserts photo of button + before/after)
5. Ensure the camping battery is in AC mode and not DC mode (TODO: Josh inserts photo of button + before/after)
6. Turn on the NUC (TODO QUESTION: do we need to do this or does it happen automatically?)

### Arm

1. Ensure that Base is powered on first, following the instructions above
2. Press the button (TODO: Josh insert image)

### Software

1. `cd prpl-tidybot`
2. `./scripts/launch.sh`
3. See base README.md for more instructions


# Power Down Process

### Software

From inside the tmux session:

1. `./hardware_tests/test_arm_retract.py` (to retract the arm to stable condition before powering off)

DO THIS FROM OUTSIDE YOUR TMUX SESSION, otherwise it will cause issues where the script kills the running tmux, which kills the script before it finishes

1. `cd prpl-tidybot`
2. `./scripts/stop_servers.sh`
3. `sudo poweroff`

### Arm

1. Press the power off button (same as the turn on button)

### Base

1. Turn off the NUC `poweroff` or `sudo poweroff`
2. Turn off the camping battery
3. Turn off the car battery's switch
4. Unplug the car battery

# Controlling the arm via controller

If you plug a usb controller into the usb ports in the arm, it will move. Don't worry about this too much since you'll do everything via software anyway.

# Controlling the Robot (via software)

Pre-Req: Ensure that you are inside the tmux session (`./scripts/launch.sh`, see README for more details)

Pre-Req: Ensure the voltage of the car battery is always between 12-13v (TOOD: reviewer double check this number). The scripts you run will print it.

1. `./hardware_tests/test_arm_ik_home.py`: this will bring the arm to home position
2. `./hardware_tests/test_arm_retract.py`: this will bring the arm to stable turn-off-able position
3. `./hardware_tests/test_gamepad_teleop.py`: this will allow you to control the base of the robot (standard drone-like controls, but you have to hold down L1 to enable them) using the gamepad (TODO: Josh insert image)

# Charging Batteries

To charge batteries, first make sure that the robot is entirely powered down via the procedure above.

## Car Battery

1. Unhitch the strap which keeps the car battery in place (TODO: Josh insert picture)
2. Ensure the car battery is unplugged
3. Take it off the robot
4. Plug it into the charger (TODO Josh insert photo)
5. The charger will show green when it is done
6. Don't leave this battery unattended for large amounts of time, as it can be dangerous. Check on it approximately hourly. We have two, so you should be able to cycle easily

## Camping Battery

With the robot close to a power outlet, unroll the wall plug attached to the camping battery and plug it in (takes ~1 hour to full). DO NOT take the camping battery off the robot