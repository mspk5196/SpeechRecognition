import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import re

class LocalNLP:
    def __init__(self, data_file="cmd_datas.csv"):
        self.data_file = data_file
        self.model = None
        self.vectorizer = None
        self.load_data()
    
    def load_data(self):
        df = pd.read_csv(self.data_file)
        self.texts = df['text'].tolist()
        self.intents = df['intent'].tolist()
        self.entities = df['entities'].tolist()

        self.vectorizer = TfidfVectorizer(lowercase=True)
        X = self.vectorizer.fit_transform(self.texts)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, self.intents)

        # Normalize keys to lowercase for robust matching
        self.entity_map = {t.lower(): e for t, e in zip(self.texts, self.entities)}
        print("NLP Model trained on local data!")

    def predict_intent(self, command_text):
        X = self.vectorizer.transform([command_text])
        intent = self.model.predict(X)[0]
        score = self.model.predict_proba(X).max()
        entities = self.extract_entities(command_text)
        # Heuristic override: pattern 'play from <time> to <time>' strongly implies playback
        lc = command_text.lower()
        if intent != 'playback' and re.search(r"play\s+from\s+\d{1,2}\s*(am|pm)\s+to\s+\d{1,2}\s*(am|pm)", lc):
            intent = 'playback'
        return {"intent": intent, "score": score, "entities": entities}

    def extract_entities(self, command_text):
        text_lc = command_text.lower()
        # Exact / containment mapping
        for base, entity_str in self.entity_map.items():
            if base in text_lc or text_lc in base:
                try:
                    entity_pairs = [p for p in entity_str.split(";") if p]
                    entities = {}
                    for pair in entity_pairs:
                        if "=" in pair:
                            k,v = pair.split("=",1)
                            entities[k.strip()] = v.strip()
                    if entities:
                        return entities
                except Exception:
                    pass

        entities: dict[str,str] = {}
        # Pattern: from 9 am to 10 am / 9am to 10am / 9 am - 10 am
        range_match = re.search(r"from\s+(\d{1,2}\s*(?:am|pm))\s+to\s+(\d{1,2}\s*(?:am|pm))", text_lc)
        if not range_match:
            range_match = re.search(r"(\d{1,2}\s*(?:am|pm))\s*(?:-|to)\s*(\d{1,2}\s*(?:am|pm))", text_lc)
        if range_match:
            entities['start_time'] = range_match.group(1).replace(" ", "")
            entities['end_time'] = range_match.group(2).replace(" ", "")

        # Single times (collect if not already captured)
        all_times = re.findall(r"\b(\d{1,2})\s*(am|pm)\b", text_lc)
        if all_times and 'start_time' not in entities:
            # First as start, last as end if multiple
            if len(all_times) == 1:
                entities['time'] = ''.join(all_times[0])
            else:
                entities['start_time'] = ''.join(all_times[0])
                entities['end_time'] = ''.join(all_times[-1])

        # Direction detection for PTZ
        if any(w in text_lc for w in ["left", "right", "up", "down"]):
            if 'direction' not in entities:
                for d in ["left","right","up","down"]:
                    if d in text_lc:
                        entities['direction'] = d
                        break

        # Location heuristic
        loc_match = re.search(r"in\s+([a-zA-Z]+)", text_lc)
        if loc_match and 'location' not in entities:
            entities['location'] = loc_match.group(1)

        # Date extraction
        # Relative words
        today_words = {"today":0, "yesterday":-1, "tomorrow":1, "day before yesterday":-2, "day after tomorrow":2}
        from datetime import date, timedelta
        for phrase, offset in sorted(today_words.items(), key=lambda x: -len(x[0])):
            if phrase in text_lc and 'date' not in entities:
                target = date.today() + timedelta(days=offset)
                entities['date'] = target.isoformat()
                break

        # Relative week/month phrases -> date ranges
        # Detect only if no explicit date already
        if 'date' not in entities:
            today = date.today()
            # Helper to first day of month
            def month_bounds(dt):
                from calendar import monthrange
                first = dt.replace(day=1)
                last = dt.replace(day=monthrange(dt.year, dt.month)[1])
                return first, last
            # Helper to week bounds (ISO week: Monday start)
            def week_bounds(dt):
                start = dt - timedelta(days=dt.weekday())
                end = start + timedelta(days=6)
                return start, end

            # Month phrases
            if 'last month' in text_lc:
                prev_month = (today.replace(day=1) - timedelta(days=1))
                first, last = month_bounds(prev_month)
                entities['date_range_start'] = first.isoformat()
                entities['date_range_end'] = last.isoformat()
            elif 'next month' in text_lc:
                next_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
                first, last = month_bounds(next_month)
                entities['date_range_start'] = first.isoformat()
                entities['date_range_end'] = last.isoformat()
            elif 'this month' in text_lc or 'current month' in text_lc:
                first, last = month_bounds(today)
                entities['date_range_start'] = first.isoformat()
                entities['date_range_end'] = last.isoformat()

            # Week phrases
            if 'last week' in text_lc:
                start, end = week_bounds(today - timedelta(days=7))
                entities['date_range_start'] = start.isoformat()
                entities['date_range_end'] = end.isoformat()
            elif 'next week' in text_lc:
                start, end = week_bounds(today + timedelta(days=7))
                entities['date_range_start'] = start.isoformat()
                entities['date_range_end'] = end.isoformat()
            elif ('this week' in text_lc or 'current week' in text_lc) and 'date_range_start' not in entities:
                start, end = week_bounds(today)
                entities['date_range_start'] = start.isoformat()
                entities['date_range_end'] = end.isoformat()

            # Specific day inside relative month: "last month on 10th", "next month on 5", "this month on 21st"
            # Only set if still no single 'date'
            if 'date' not in entities:
                m_rel_day = re.search(r"(last|next|this|current)\s+month\s+on\s+(\d{1,2})(?:st|nd|rd|th)?\b", text_lc)
                if m_rel_day:
                    rel = m_rel_day.group(1)
                    day_num = int(m_rel_day.group(2))
                    # derive target month
                    if rel == 'last':
                        target_month = (today.replace(day=1) - timedelta(days=1))
                    elif rel == 'next':
                        target_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
                    else:  # this/current
                        target_month = today
                    # clamp day within month
                    from calendar import monthrange as _mr
                    max_day = _mr(target_month.year, target_month.month)[1]
                    if day_num <= max_day:
                        try:
                            entities['date'] = target_month.replace(day=day_num).isoformat()
                            # If a month range was previously added and now a specific day chosen, we keep both; command synthesis prefers DATE over DATE_RANGE.
                        except Exception:
                            pass

        # Explicit date patterns (dd/mm/yyyy, dd-mm-yyyy, dd month yyyy, month dd yyyy)
        # 1) Numeric (allow yyyy optional)
        m_num = re.search(r"\b(\d{1,2})[\/-](\d{1,2})(?:[\/-](\d{2,4}))?\b", text_lc)
        if m_num and 'date' not in entities:
            d = int(m_num.group(1))
            m = int(m_num.group(2))
            y = m_num.group(3)
            if y:
                y = int(y if len(y) == 4 else ('20'+y))
            else:
                y = date.today().year
            try:
                entities['date'] = date(y,m,d).isoformat()
            except Exception:
                pass

        # 2) Month name forms
        if 'date' not in entities:
            month_map = {m.lower():i for i,m in enumerate(['January','February','March','April','May','June','July','August','September','October','November','December'], start=1)}
            m_name = re.search(r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*(?:\s+(\d{4}))?\b", text_lc)
            if m_name:
                d = int(m_name.group(1))
                mon_token = m_name.group(2)
                y = m_name.group(3)
                if y:
                    y = int(y)
                else:
                    y = date.today().year
                # Normalize month token
                norm = mon_token[:3]
                month_alias = {
                    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                    'jul':7,'aug':8,'sep':9,'sept':9,'oct':10,'nov':11,'dec':12
                }
                m = month_alias.get(norm, None)
                if m:
                    try:
                        entities['date'] = date(y,m,d).isoformat()
                    except Exception:
                        pass

        # 3) Month name first (march 5 2025)
        if 'date' not in entities:
            m_name2 = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*\s+(\d{1,2})(?:,?\s*(\d{4}))?\b", text_lc)
            if m_name2:
                mon_token = m_name2.group(1)
                d = int(m_name2.group(2))
                y = m_name2.group(3)
                if y:
                    y = int(y)
                else:
                    y = date.today().year
                month_alias = {
                    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                    'jul':7,'aug':8,'sep':9,'sept':9,'oct':10,'nov':11,'dec':12
                }
                m = month_alias.get(mon_token[:3], None)
                if m:
                    try:
                        entities['date'] = date(y,m,d).isoformat()
                    except Exception:
                        pass

        return entities
    
if __name__ == "__main__":
    nlp = LocalNLP()
    commands = [
        "Show playback from 9 am to 10 am",
        "Move camera to left",
        "கேமரா இடது பக்கம் நகர்த்து"
    ]
    for cmd in commands:
        res = nlp.predict_intent(cmd)
        print(f"Command: {cmd}\nIntent: {res['intent']}\nEntities: {res['entities']}\n")
