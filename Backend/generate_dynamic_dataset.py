import csv, sys, random, datetime as dt

# Configuration
OUTPUT_FILE = 'generated_playback_dataset.csv'
SEED = 42
random.seed(SEED)

# Time windows to sample (hour ranges) 24h pairs ensuring start < end
HOUR_RANGES = [(h, h+1) for h in range(0,23)] + [(8,10),(9,12),(10,13),(13,15),(15,18),(18,21)]
HOUR_RANGES = [r for r in HOUR_RANGES if r[1] <= 23]

SINGLE_STARTS = list(range(0,24))

DATE_MODES = [
    ('today', 0),
    ('yesterday', -1),
    ('day_before_yesterday', -2),
    ('none', None),
]

CAMERAS = ['', 'camera 1', 'camera 2', 'camera 3', 'cam 4', 'channel 5']

TEMPLATES_RANGE = [
    'from {st_label} to {et_label}',
    'from {st_label} till {et_label}',
    'between {st_label} and {et_label}',
    'show playback from {st_label} to {et_label}',
    'playback {st_label}-{et_label}',
    'video {st_label} to {et_label}',
    '{st_label} to {et_label} footage',
    'need recording {st_label} to {et_label}',
    'retrieve {st_label} to {et_label}',
]

TEMPLATES_SINGLE = [
    'from {st_label}',
    'at {st_label}',
    'around {st_label}',
    'show playback at {st_label}',
    'need recording from {st_label}',
]

PART_OF_DAY = [
    ('morning', range(5,12)),
    ('afternoon', range(12,17)),
    ('evening', range(17,21)),
    ('night', range(21,24)),
]

def hour_label(h):
    # Return both 12h and plain forms randomly
    ampm = 'am' if h < 12 else 'pm'
    h12 = h % 12
    if h12 == 0: h12 = 12
    style = random.choice(['12h','24h','phrase'])
    if style == '12h':
        return f"{h12}{ampm}"
    if style == '24h':
        return f"{h:02d}:00"
    # phrase with part of day when possible
    pod = None
    for label, rng in PART_OF_DAY:
        if h in rng:
            pod = label
            break
    if pod:
        return f"{h12} in the {pod}"
    return f"{h12}{ampm}"


def build_entities(start_h, end_h=None, date_mode=('none',None)):
    today = dt.date.today()
    date_key, offset = date_mode
    ent = {}
    ent['start_time'] = f"{start_h:02d}:00"
    if end_h is not None:
        ent['end_time'] = f"{end_h:02d}:00"
    if offset is not None:
        target = today + dt.timedelta(days=offset)
        ent['date'] = target.isoformat()
    return ';'.join([f"{k}={v}" for k,v in ent.items()])


def generate_rows(n_range=300, n_single=120):
    rows = []
    # Ranges
    for _ in range(n_range):
        sh, eh = random.choice(HOUR_RANGES)
        tmpl = random.choice(TEMPLATES_RANGE)
        st_label = hour_label(sh)
        et_label = hour_label(eh)
        date_mode = random.choice(DATE_MODES)
        cam = random.choice(CAMERAS)
        text = tmpl.format(st_label=st_label, et_label=et_label)
        if date_mode[0] == 'today':
            text = 'today ' + text
        elif date_mode[0] == 'yesterday':
            text = 'yesterday ' + text
        elif date_mode[0] == 'day_before_yesterday':
            text = 'day before yesterday ' + text
        if cam:
            text = text + ' ' + cam
        entities = build_entities(sh, eh, date_mode)
        rows.append((text.strip(), 'playback', entities))
    # Singles
    for _ in range(n_single):
        sh = random.choice(SINGLE_STARTS)
        tmpl = random.choice(TEMPLATES_SINGLE)
        st_label = hour_label(sh)
        date_mode = random.choice(DATE_MODES)
        cam = random.choice(CAMERAS)
        text = tmpl.format(st_label=st_label)
        if date_mode[0] == 'today':
            text = 'today ' + text
        elif date_mode[0] == 'yesterday':
            text = 'yesterday ' + text
        elif date_mode[0] == 'day_before_yesterday':
            text = 'day before yesterday ' + text
        if cam:
            text = text + ' ' + cam
        entities = build_entities(sh, None, date_mode)
        rows.append((text.strip(), 'playback', entities))
    return rows


def main():
    rows = generate_rows()
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['text','intent','entities'])
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
