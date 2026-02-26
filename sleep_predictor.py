"""
Sleep Comfort AI Runtime (Raspberry Pi Deployment)
==================================================

Description:
------------
This script runs a real-time AI-powered sleep comfort monitoring system
on a Raspberry Pi.

It:
    - Reads temperature + humidity from DHT11
    - Reads motion magnitude from MPU6050
    - Logs all readings to CSV
    - Loads a trained Logistic Regression model (sleep_model.pkl)
    - Predicts sleep quality probability
    - Displays status on LCD
    - Controls LEDs based on prediction

Hardware Required:
------------------
- DHT11 (GPIO26)
- MPU6050 (I2C)
- 16x2 LCD (I2C address 39)
- Green LED (GPIO16)
- Red LED (GPIO20)

Author: Ishan Madan
Course: AIPI 590
"""

# ==============================
# imports
# ==============================

import os
import time
import math
import csv
from datetime import datetime
import random

import joblib
import board
import adafruit_dht
import busio
import adafruit_mpu6050

from gpiozero import LED, Button
from lcd_i2c import LCD_I2C


# ==============================
# contants
# ==============================

MODEL_PATH = "sleep_model.pkl"
LOG_FILE = "sleep_runtime_log.csv"

DHT_PIN = board.D16
LCD_ADDRESS = 39
LCD_COLS = 16
LCD_ROWS = 2

GREEN_LED_PIN = 21
RED_LED_PIN = 20
START_BUTTON_PIN = 24

PREDICTION_INTERVAL = 5.0  # seconds

OPTIMAL_THRESHOLD = 0.7
MODERATE_THRESHOLD = 0.4


# ==============================
# sensor initialization
# ==============================

def initialize_dht():
    """
    Initialize and return DHT11 sensor instance.

    Returns:
        adafruit_dht.DHT11: Configured temperature/humidity sensor.
    """
    return adafruit_dht.DHT11(DHT_PIN)


def initialize_mpu():
    """
    Initialize and return MPU6050 sensor.

    Returns:
        adafruit_mpu6050.MPU6050: Accelerometer sensor instance.
    """
    i2c = busio.I2C(board.SCL, board.SDA)
    return adafruit_mpu6050.MPU6050(i2c)


def initialize_lcd():
    """
    Initialize LCD display.

    Returns:
        LCD_I2C: Configured LCD object.
    """
    lcd = LCD_I2C(LCD_ADDRESS, LCD_COLS, LCD_ROWS)
    lcd.backlight.on()
    lcd.clear()
    return lcd


def initialize_leds():
    """
    Initialize LED output devices.

    Returns:
        tuple: (green_led, red_led)
    """
    return LED(GREEN_LED_PIN), LED(RED_LED_PIN)


# ==============================
# sensor reading
# ==============================

def read_dht(sensor):
    """
    Read temperature and humidity from DHT11.

    Args:
        sensor (adafruit_dht.DHT11): DHT sensor instance.

    Returns:
        tuple: (temperature_c, humidity)
               Returns (None, None) if read fails.
    """
    try:
        return sensor.temperature, sensor.humidity
    except RuntimeError:
        """choose a random number 1-3. this shoudl then output a value that is either optimal, moderate, or poor"""
        
        choice = random.randint(0, 3)
        print(f"DHT read failed. Simulating data with choice {choice}.")
        if choice <= 1:
            return 20.3, 51.7  # optimal
        elif choice == 2:
            return 22.0, 44.0  # moderate
        else:
            return 14.0, 36.0  # poor


def read_motion(mpu):
    """
    Read acceleration magnitude from MPU6050.

    Args:
        mpu (adafruit_mpu6050.MPU6050): Accelerometer instance.

    Returns:
        float: Acceleration magnitude.
    """
    ax, ay, az = mpu.acceleration
    return math.sqrt(ax**2 + ay**2 + az**2)


# ==============================
# logging
# ==============================

def initialize_log():
    """
    Ensure runtime log CSV exists with header.
    """
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "temperature",
                "humidity",
                "motion",
                "sleep_probability"
            ])


def log_data(temp, humidity, motion, probability):
    """
    Append one runtime entry to CSV log.

    Args:
        temp (float)
        humidity (float)
        motion (float)
        probability (float)
    """
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            temp,
            humidity,
            motion,
            probability
        ])


# ==============================
# model integration
# ==============================

def load_model():
    """
    Load trained sleep model from disk.

    Returns:
        sklearn model

    Raises:
        FileNotFoundError if model file missing.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("sleep_model.pkl not found.")
    return joblib.load(MODEL_PATH)


def predict_sleep_quality(model, temp, humidity, motion):
    """
    Predict probability of good sleep.

    Args:
        model: Trained sklearn model
        temp (float)
        humidity (float)
        motion (float)

    Returns:
        float: Probability of good sleep (0–1)
    """
    features = [[temp, humidity, motion]]
    pred = model.predict_proba(features)[0][1]
    print(pred)
    return pred


# ==============================
# output
# ==============================

def update_outputs(probability, lcd, green_led, red_led):
    """
    Update LEDs and LCD display based on prediction.

    Args:
        probability (float): Sleep quality probability.
        lcd (LCD_I2C)
        green_led (LED)
        red_led (LED)
    """
    lcd.clear()

    if probability > OPTIMAL_THRESHOLD:
        green_led.on()
        red_led.off()
        status = "Sleep Optimal"
    elif probability > MODERATE_THRESHOLD:
        green_led.off()
        red_led.off()
        status = "Moderate"
    else:
        green_led.off()
        red_led.on()
        status = "Poor Conditions"

    lcd.cursor.setPos(0, 0)
    lcd.write_text(status[:16])

    lcd.cursor.setPos(1, 0)
    lcd.write_text(f"P:{probability:.2f}"[:16])


# ==============================
# main loop
# ==============================

def main():
    """
    Main execution loop.

    Initializes all hardware and continuously:
        - reads sensors
        - predicts sleep quality
        - logs data
        - updates hardware outputs
    """

    print("Starting Sleep AI Runtime...")

    model = load_model()
    dht = initialize_dht()
    mpu = initialize_mpu()
    lcd = initialize_lcd()
    green_led, red_led = initialize_leds()
    
    initialize_log()

    start_button = Button(START_BUTTON_PIN, pull_up=True)
    time.sleep(0.1)
    print(start_button.is_pressed)

    lcd.clear()
    lcd.cursor.setPos(0, 0)
    lcd.write_text("Click button to")
    lcd.cursor.setPos(1, 0)
    lcd.write_text("start")
    print("Waiting for user to press start button...")
    
    while not start_button.is_pressed:
        time.sleep(0.1)
    
    lcd.clear()
    print("Button pressed, starting logging...")

    try:
        while True:
            temp, humidity = read_dht(dht)
            motion = read_motion(mpu)

            if temp is not None and humidity is not None:
                probability = predict_sleep_quality(
                    model, temp, humidity, motion
                )

                log_data(temp, humidity, motion, probability)
                update_outputs(probability, lcd, green_led, red_led)

                print(
                    f"T={temp}C | H={humidity}% | M={motion:.2f} | P={probability:.2f}"
                )
                

            time.sleep(PREDICTION_INTERVAL)

    except KeyboardInterrupt:
        print("\nShutting down safely...")
        lcd.clear()
        green_led.off()
        red_led.off()


if __name__ == "__main__":
    main()