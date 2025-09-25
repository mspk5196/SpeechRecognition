import csv
from random import choice, randint
from datetime import datetime, timedelta

english_locations = ["garden", "parking", "main hall", "lobby", "front building"]
tamil_locations = ["முகப்பு தோட்டம்", "வாகன நிறுத்த இடம்", "மெயின் ஹால்", "லாபி", "முன்னணி மாளிகை"]
directions = ["left", "right", "up", "down"]
actions = ["start", "stop", "zoom_in", "zoom_out"]
time_phrases_en = ["today", "yesterday", "day before yesterday", "last week", "last month"]
time_phrases_ta = ["இன்று", "நேற்று", "நேற்று முன் நாள்", "கடந்த வாரம்", "கடந்த மாதம்"]

intents = ["playback", "ptz", "motion_check", "recording"]

commands = []

# Helper to generate random time strings
def random_time_range():
    start = randint(0, 23)
    end = (start + randint(1, 3)) % 24
    return f"{start}:00 to {end}:00", f"{start} மணி முதல் {end} மணி வரை"

# Generate playback commands with specific and relative dates
for i in range(60):
    loc_en = choice(english_locations)
    loc_ta = choice(tamil_locations)
    date_choice = choice(time_phrases_en)
    date_choice_ta = choice(time_phrases_ta)
    time_text_en, time_text_ta = random_time_range()
    
    # English
    commands.append([
        f"Show playback from {time_text_en} {date_choice}",
        "playback",
        f"start_time={time_text_en.split(' to ')[0]};end_time={time_text_en.split(' to ')[1]};date={date_choice}"
    ])
    
    # Tamil
    commands.append([
        f"{date_choice_ta} {time_text_ta} பதிவு காண்பி",
        "playback",
        f"start_time={time_text_ta.split(' முதல் ')[0]};end_time={time_text_ta.split(' வரை')[0]};date={date_choice_ta}"
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

# Write to CSV
with open("cmd_datas.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text","intent","entities"])
    writer.writerows(commands)

print(f"Generated {len(commands)} commands in cmd_datas.csv")
