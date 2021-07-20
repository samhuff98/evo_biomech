import RPi.GPIO as GPIO
from gpiozero import Button
import time
import subprocess, os
import signal

def ticcmd(*args):
    return subprocess.check_output(['ticcmd'] + list(args))

x_value = 0
GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO_switch = 11
GPIO_LED_red = 13
GPIO_LED_green = 15
GPIO.setup(GPIO_switch, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(GPIO_LED_red, GPIO.OUT)
GPIO.setup(GPIO_LED_green, GPIO.OUT)

try:
    runs = 0
    while True:
        if GPIO.input(GPIO_switch)==False:
            rpistr = "python /home/pi/Documents/evo_biomech/treadmill.py"
            p=subprocess.Popen(rpistr, shell=True, preexec_fn=os.setsid)
            GPIO.output(GPIO_LED_red, False)
            GPIO.output(GPIO_LED_green, True)
            time.sleep(0.5)
            GPIO.output(GPIO_LED_green, False)
            time.sleep(3.5)
            GPIO.output(GPIO_LED_green, True)
            runs+=1
            while GPIO.input(GPIO_switch)==False:
                time.sleep(0.01)
        if GPIO.input(GPIO_switch)==True and runs>=1:
            os.killpg(p.pid, signal.SIGTERM)
            GPIO.output(GPIO_LED_red, True)
            GPIO.output(GPIO_LED_green, False)
            ticcmd('--enter-safe-start')
            ticcmd('--deenergize')
            while GPIO.input(GPIO_switch)==True:
                time.sleep(0.01)
        if GPIO.input(GPIO_switch)==True and runs==0:
            GPIO.output(GPIO_LED_red, True)
            GPIO.output(GPIO_LED_green, False)
            while GPIO.input(GPIO_switch)==True:
                time.sleep(0.01)

except KeyboardInterrupt:
    os.killpg(p.pid, signal.SIGTERM)
    print("Program stopped")
    ticcmd('--enter-safe-start')
    ticcmd('--deenergize')
    GPIO.cleanup()
    exit(0)