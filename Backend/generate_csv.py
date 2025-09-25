import csv
from random import choice, randint
from datetime import datetime, timedelta

# Locations
english_locations = ["garden", "parking", "main hall", "lobby", "front building"]
tamil_locations = ["முகப்பு தோட்டம்", "வாகன நிறுத்த இடம்", "மெயின் ஹால்", "லாபி", "முன்னணி மாளிகை"]

# Directions for PTZ
directions = ["left", "right", "up", "down"]

# Relative days
time_phrases_en = ["today", "yesterday", "day before yesterday", "last week", "last month"]
time_phrases_ta = ["இன்று", "நேற்று", "நேற்று முன் நாள்", "கடந்த வாரம்", "கடந்த மாதம்"]

# Weekdays for "this week" and "last week"
weekdays_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekdays_ta = ["திங்கட்கிழமை", "செவ்வாய்க்கிழமை", "புதன்கிழமை", "வியாழக்கிழமை", "வெள்ளிக்கிழமை", "சனிக்கிழமை", "ஞாயிற்றுக்கிழமை"]

# Intents
intents = ["playback", "ptz", "motion_check", "recording"]

commands = []

# Helper to generate random time ranges
def random_time_range():
    start = randint(0, 23)
    end = (start + randint(1, 3)) % 24
    return f"{start}:00 to {end}:00", f"{start} மணி முதல் {end} மணி வரை"

# Generate playback commands with specific dates, relative days, and weekdays
for i in range(100):
    loc_en = choice(english_locations)
    loc_ta = choice(tamil_locations)

    # Absolute date
    date_obj = datetime.now() - timedelta(days=randint(0, 60))
    date_str_en = date_obj.strftime("%Y-%m-%d")
    date_str_ta = date_obj.strftime("%Y-%m-%d")  # Could add Tamil formatting if needed

    # Relative day
    rel_day_en = choice(time_phrases_en)
    rel_day_ta = choice(time_phrases_ta)

    # Weekday
    weekday_idx = randint(0, 6)
    weekday_en = weekdays_en[weekday_idx]
    weekday_ta = weekdays_ta[weekday_idx]
    week_type = choice(["this week", "last week"])
    week_type_ta = "இந்த வாரம்" if week_type == "this week" else "கடந்த வாரம்"

    # Time range
    time_text_en, time_text_ta = random_time_range()

    # Absolute date example
    commands.append([
        f"Show playback on {date_str_en} from {time_text_en}",
        "playback",
        f"start_time={time_text_en.split(' to ')[0]};end_time={time_text_en.split(' to ')[1]};date={date_str_en}"
    ])
    commands.append([
        f"{date_str_ta} {time_text_ta} பதிவு காண்பி",
        "playback",
        f"start_time={time_text_ta.split(' முதல் ')[0]};end_time={time_text_ta.split(' வரை')[0]};date={date_str_ta}"
    ])

    # Relative day example
    commands.append([
        f"Show playback from {time_text_en} {rel_day_en}",
        "playback",
        f"start_time={time_text_en.split(' to ')[0]};end_time={time_text_en.split(' to ')[1]};date={rel_day_en}"
    ])
    commands.append([
        f"{rel_day_ta} {time_text_ta} பதிவு காண்பி",
        "playback",
        f"start_time={time_text_ta.split(' முதல் ')[0]};end_time={time_text_ta.split(' வரை')[0]};date={rel_day_ta}"
    ])

    # Weekday example
    commands.append([
        f"Show playback on {weekday_en} ({week_type}) from {time_text_en}",
        "playback",
        f"start_time={time_text_en.split(' to ')[0]};end_time={time_text_en.split(' to ')[1]};date={weekday_en} {week_type}"
    ])
    commands.append([
        f"{week_type_ta} {weekday_ta} {time_text_ta} பதிவு காண்பி",
        "playback",
        f"start_time={time_text_ta.split(' முதல் ')[0]};end_time={time_text_ta.split(' வரை')[0]};date={weekday_ta} {week_type_ta}"
    ])

# PTZ commands
for i in range(30):
    dir_en = choice(directions)
    dir_ta_map = {"left":"இடது", "right":"வலது", "up":"மேல்", "down":"கீழ்"}
    dir_ta = dir_ta_map[dir_en]
    loc_en = choice(english_locations)
    loc_ta = choice(tamil_locations)
    commands.append([f"Move camera {dir_en} in {loc_en}", "ptz", f"direction={dir_en};location={loc_en}"])
    commands.append([f"கேமரா {dir_ta} {loc_ta} நகர்த்து", "ptz", f"direction={dir_en};location={loc_ta}"])

# Motion check commands
for i in range(30):
    loc_en = choice(english_locations)
    loc_ta = choice(tamil_locations)
    commands.append([f"Any movement in the {loc_en}?", "motion_check", f"location={loc_en}"])
    commands.append([f"{loc_ta} இடத்தில் இயக்கம் ஏதும் உள்ளதா?", "motion_check", f"location={loc_ta}"])

# Recording commands
for i in range(30):
    action_en = choice(["start","stop"])
    loc_en = choice(english_locations)
    loc_ta = choice(tamil_locations)
    action_ta_map = {"start":"தொடங்கு", "stop":"நிறுத்து"}
    action_ta = action_ta_map[action_en]
    commands.append([f"{action_en.capitalize()} recording in {loc_en}", "recording", f"action={action_en};location={loc_en}"])
    commands.append([f"{loc_ta} இடத்தில் பதிவு {action_ta}", "recording", f"action={action_en};location={loc_ta}"])

# Write CSV
with open("commands_full_dates.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text","intent","entities"])
    writer.writerows(commands)

print(f"Generated {len(commands)} commands in commands_full_dates.csv")
