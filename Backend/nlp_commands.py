import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import os
from typing import List, Dict, Optional, Union

class LocalNLP:
    """Local NLP component with optional incremental (buffered) self-learning.

    Self-learning strategy (conservative to avoid model drift):
    - During prediction, high-confidence examples (>= confidence_threshold) are buffered.
    - When buffer reaches buffer_max (or flush is triggered manually) they are appended
      to a separate user data file and the model is retrained from scratch on
      (original + user) datasets to keep vectorizer vocabulary in sync.
    - A public learn_example() method allows explicit supervised additions (e.g. when
      the UI lets a user correct an intent). These can optionally retrain immediately.

    Note: Training on self-predicted labels can reinforce mistakes. Keep a high
    threshold and prefer explicit user-confirmed feedback where possible.
    """

    def __init__(self,
                 data_file: str = "cmd_datas.csv",
                 user_data_file: Optional[str] = None,
                 auto_learn: bool = True,
                 confidence_threshold: float = 0.9,
                 buffer_max: int = 20):
        self.data_file = data_file
        # Derive user data file path if not provided
        if user_data_file is None:
            root, ext = os.path.splitext(self.data_file)
            user_data_file = f"{root}_user{ext or '.csv'}"
        self.user_data_file = user_data_file

        self.auto_learn = auto_learn
        self.confidence_threshold = confidence_threshold
        self.buffer_max = buffer_max
        self._buffer: List[Dict[str, str]] = []  # each: {'text','intent','entities'}

        self.model: Optional[LogisticRegression] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.texts: List[str] = []
        self.intents: List[str] = []
        self.entities: List[str] = []
        self.entity_map: Dict[str, str] = {}
        self.load_data()

    def load_data(self):
        """Load base + user data, (re)train vectorizer & model."""
        if not os.path.isfile(self.data_file):
            raise FileNotFoundError(f"Base data file not found: {self.data_file}")

        df_base = pd.read_csv(self.data_file)
        frames = [df_base]
        if os.path.isfile(self.user_data_file):
            try:
                df_user = pd.read_csv(self.user_data_file)
                frames.append(df_user)
            except Exception:
                pass
        df = pd.concat(frames, ignore_index=True)

        # Deduplicate by text to avoid overweighting repeated items
        df = df.drop_duplicates(subset=['text']).reset_index(drop=True)

        self.texts = df['text'].tolist()
        self.intents = df['intent'].tolist()
        self.entities = df['entities'].tolist()

        self.vectorizer = TfidfVectorizer(lowercase=True)
        X = self.vectorizer.fit_transform(self.texts)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, self.intents)

        # Normalize keys to lowercase for robust matching
        self.entity_map = {t.lower(): e for t, e in zip(self.texts, self.entities)}
        print(f"NLP Model trained on {len(self.texts)} examples (base + user).")

    # ---------------------------------------------------------------------
    # Public API for incremental learning
    # ---------------------------------------------------------------------
    def learn_example(self,
                      text: str,
                      intent: str,
                      entities: Union[Dict[str, str], str, None] = None,
                      retrain: bool = True):
        """Add a supervised example (e.g. after user correction).

        entities can be a dict (will be serialized as k=v;) or a preformatted string.
        If retrain=True the model is reloaded immediately; else example is appended
        to user data file for later retraining.
        """
        if not text or not intent:
            return False
        ent_str = self._serialize_entities(entities)
        self._append_user_example(text, intent, ent_str)
        if retrain:
            self.load_data()
        return True

    def flush_auto_learning(self):
        """Persist buffered auto-learn examples and retrain."""
        if not self._buffer:
            return 0
        flushed = 0
        for item in self._buffer:
            self._append_user_example(item['text'], item['intent'], item['entities'])
            flushed += 1
        self._buffer.clear()
        self.load_data()
        return flushed

    def toggle_auto_learn(self, enabled: bool):
        self.auto_learn = enabled

    def retrain(self):
        """Explicit full retrain (rebuild vectorizer/model)."""
        self.load_data()

    # ------------------------------------------------------------------
    # Internal helpers for learning persistence
    # ------------------------------------------------------------------
    def _append_user_example(self, text: str, intent: str, entities: str):
        header_needed = not os.path.isfile(self.user_data_file)
        with open(self.user_data_file, 'a', encoding='utf-8') as f:
            if header_needed:
                f.write('text,intent,entities\n')
            # Escape newlines / commas minimally by replacing commas in entities
            safe_entities = entities.replace('\n', ' ').replace(',', ' ')
            safe_text = text.replace('\n', ' ').replace(',', ' ')
            safe_intent = intent.replace('\n', ' ').replace(',', ' ')
            f.write(f"{safe_text},{safe_intent},{safe_entities}\n")

    def _serialize_entities(self, entities: Union[Dict[str, str], str, None]) -> str:
        if entities is None:
            return ''
        if isinstance(entities, str):
            return entities
        parts = []
        for k, v in entities.items():
            parts.append(f"{k}={v}")
        return ';'.join(parts)

    def _maybe_auto_learn(self, text: str, intent: str, score: float, entities: Dict[str, str]):
        if not self.auto_learn:
            return
        if score < self.confidence_threshold:
            return
        # Avoid duplicate texts in buffer
        if any(b['text'].lower() == text.lower() for b in self._buffer):
            return
        serialized = self._serialize_entities(entities)
        self._buffer.append({
            'text': text,
            'intent': intent,
            'entities': serialized
        })
        if len(self._buffer) >= self.buffer_max:
            self.flush_auto_learning()

    def predict_intent(self, command_text):
        X = self.vectorizer.transform([command_text])
        intent = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        score = float(proba.max())
        entities = self.extract_entities(command_text)
        # Heuristic override: pattern 'play from <time> to <time>' strongly implies playback
        lc = command_text.lower()
        if intent != 'playback' and re.search(r"play\s+from\s+\d{1,2}\s*(am|pm)\s+to\s+\d{1,2}\s*(am|pm)", lc):
            intent = 'playback'
        # Attempt auto-learning if enabled
        self._maybe_auto_learn(command_text, intent, score, entities)
        return {"intent": intent, "score": score, "entities": entities}

    def extract_entities(self, command_text):
        text_lc = command_text.lower()
        # Exact / containment mapping (collect but don't early-return so we can refine dates/times)
        collected = {}
        for base, entity_str in self.entity_map.items():
            if base in text_lc or text_lc in base:
                try:
                    entity_pairs = [p for p in entity_str.split(";") if p]
                    for pair in entity_pairs:
                        if "=" in pair:
                            k,v = pair.split("=",1)
                            if k.strip() and v.strip():
                                collected[k.strip()] = v.strip()
                except Exception:
                    pass

        entities: dict[str,str] = collected.copy()
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

        # Camera / channel selection
        if 'camera' not in entities:
            # Numeric forms: camera 1, cam 02, channel 3, ch 4
            cam_match = re.search(r"\b(cam(?:era)?|channel|ch)\s*(\d{1,3})\b", text_lc)
            if cam_match:
                entities['camera'] = str(int(cam_match.group(2)))  # normalize remove leading zeros
            else:
                # Word number forms: camera one/two/three ... up to twelve
                word_nums = {
                    'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12
                }
                cam_word = re.search(r"\b(cam(?:era)?|channel|ch)\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", text_lc)
                if cam_word:
                    entities['camera'] = str(word_nums[cam_word.group(2)])

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

            # Multi-day span inside relative month (process before single day):
            if 'date' not in entities:
                # Forms: "last month 10th to 12th", "last month 10 to 12", reversed: "10th to 12th last month"
                span_patterns = [
                    r"(last|next|this|current)\s+month\s+(\d{1,2})(?:st|nd|rd|th)?\s*(?:to|-|through)\s*(\d{1,2})(?:st|nd|rd|th)?\b",
                    r"(\d{1,2})(?:st|nd|rd|th)?\s*(?:to|-|through)\s*(\d{1,2})(?:st|nd|rd|th)?\s+(last|next|this|current)\s+month"
                ]
                for pat in span_patterns:
                    m_span = re.search(pat, text_lc)
                    if m_span:
                        if pat.startswith('(last'):
                            rel = m_span.group(1); d1 = int(m_span.group(2)); d2 = int(m_span.group(3))
                        else:
                            d1 = int(m_span.group(1)); d2 = int(m_span.group(2)); rel = m_span.group(3)
                        if d2 < d1:
                            d1, d2 = d2, d1
                        if rel == 'last':
                            base_month = (today.replace(day=1) - timedelta(days=1))
                        elif rel == 'next':
                            base_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
                        else:
                            base_month = today
                        from calendar import monthrange as _mr
                        max_day = _mr(base_month.year, base_month.month)[1]
                        if d1 <= max_day and d2 <= max_day:
                            try:
                                start_date = base_month.replace(day=d1).isoformat()
                                end_date = base_month.replace(day=d2).isoformat()
                                entities['date_range_start'] = start_date
                                entities['date_range_end'] = end_date
                            except Exception:
                                pass
                        break

            # Specific day inside relative month: variations
            if 'date' not in entities:
                rel_day_patterns = [
                    r"(last|next|this|current)\s+month\s+on\s+(\d{1,2})(?:st|nd|rd|th)?\b",
                    r"(last|next|this|current)\s+month\s+(\d{1,2})(?:st|nd|rd|th)?\b",  # without 'on'
                    r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(last|next|this|current)\s+month"  # reversed
                ]
                for pat in rel_day_patterns:
                    m_rel_day = re.search(pat, text_lc)
                    if m_rel_day:
                        if pat.startswith('(last'):
                            rel = m_rel_day.group(1); day_num = int(m_rel_day.group(2))
                        elif pat.startswith('(\d'):
                            day_num = int(m_rel_day.group(1)); rel = m_rel_day.group(2)
                        else:
                            rel = m_rel_day.group(1); day_num = int(m_rel_day.group(2))
                        if rel == 'last':
                            target_month = (today.replace(day=1) - timedelta(days=1))
                        elif rel == 'next':
                            target_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
                        else:
                            target_month = today
                        from calendar import monthrange as _mr
                        max_day = _mr(target_month.year, target_month.month)[1]
                        if day_num <= max_day:
                            try:
                                specific = target_month.replace(day=day_num).isoformat()
                                entities['date'] = specific
                                entities['date_range_start'] = specific
                                entities['date_range_end'] = specific
                            except Exception:
                                pass
                        break

            # Tamil month relative: கடந்த மாதம் / இந்த மாதம் / அடுத்த மாதம் (+ day or span)
            if 'date' not in entities:
                # Span: கடந்த மாதம் 10ம் தேதி முதல் 12ம் தேதி (muthal ... ) OR using to/-
                tamil_span = re.search(r"(கடந்த|இந்த|அடுத்த)\s+மாதம்\s+(\d{1,2})(?:ம்)?\s*தேதி\s*(?:முதல்|to|-)?\s*(\d{1,2})(?:ம்)?\s*தேதி", text_lc)
                tamil_single = None
                rel_map = {'கடந்த':'last','இந்த':'this','அடுத்த':'next'}
                if tamil_span:
                    rel_t = tamil_span.group(1); d1 = int(tamil_span.group(2)); d2 = int(tamil_span.group(3))
                    if d2 < d1: d1, d2 = d2, d1
                    rel = rel_map.get(rel_t, 'this')
                    if rel == 'last':
                        base_month = (today.replace(day=1) - timedelta(days=1))
                    elif rel == 'next':
                        base_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
                    else:
                        base_month = today
                    from calendar import monthrange as _mr
                    max_day = _mr(base_month.year, base_month.month)[1]
                    if d1 <= max_day and d2 <= max_day:
                        try:
                            entities['date_range_start'] = base_month.replace(day=d1).isoformat()
                            entities['date_range_end'] = base_month.replace(day=d2).isoformat()
                        except Exception:
                            pass
                else:
                    tamil_single = re.search(r"(கடந்த|இந்த|அடுத்த)\s+மாதம்\s+(\d{1,2})(?:ம்)?\s*தேதி", text_lc)
                    if tamil_single:
                        rel_t = tamil_single.group(1); d = int(tamil_single.group(2))
                        rel = rel_map.get(rel_t, 'this')
                        if rel == 'last':
                            base_month = (today.replace(day=1) - timedelta(days=1))
                        elif rel == 'next':
                            base_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
                        else:
                            base_month = today
                        from calendar import monthrange as _mr
                        max_day = _mr(base_month.year, base_month.month)[1]
                        if d <= max_day:
                            try:
                                specific = base_month.replace(day=d).isoformat()
                                entities['date'] = specific
                                entities['date_range_start'] = specific
                                entities['date_range_end'] = specific
                            except Exception:
                                pass

            # Weekday handling (English + Tamil). Only if still no date.
            if 'date' not in entities:
                # English weekdays
                weekdays = {
                    'monday':0,'tuesday':1,'wednesday':2,'thursday':3,'friday':4,'saturday':5,'sunday':6
                }
                # Tamil approximate transliterations / words (can expand):
                tamil_week = {
                    'திங்கள்':0,'செவ்வாய்':1,'புதன்':2,'வியாழன்':3,'வெள்ளி':4,'சனி':5,'ஞாயிறு':6,
                    'ஞாயிறு.':6
                }
                # Pattern examples: 'on monday', 'next tuesday', 'last wednesday', 'this friday'
                m_wd = re.search(r"\b(on\s+)?(last|next|this|current)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text_lc)
                m_wd_ta = None
                if not m_wd:
                    # Tamil pattern: '(கடந்த|அடுத்த|இந்த) திங்கள்'
                    m_wd_ta = re.search(r"\b(கடந்த|அடுத்த|இந்த)?\s*(திங்கள்|செவ்வாய்|புதன்|வியாழன்|வெள்ளி|சனி|ஞாயிறு)\b", text_lc)
                target_date = None
                if m_wd:
                    rel = m_wd.group(2) or 'this'
                    wd = m_wd.group(3)
                    base = today
                    # compute start of current week (Monday start)
                    week_start = base - timedelta(days=base.weekday())
                    if rel == 'last':
                        week_start = week_start - timedelta(days=7)
                    elif rel == 'next':
                        week_start = week_start + timedelta(days=7)
                    idx = weekdays[wd]
                    target_date = week_start + timedelta(days=idx)
                elif m_wd_ta:
                    rel_word = m_wd_ta.group(1) or 'இந்த'
                    wd_ta = m_wd_ta.group(2)
                    rel_map_ta = {'கடந்த':'last','அடுத்த':'next','இந்த':'this'}
                    rel = rel_map_ta.get(rel_word, 'this')
                    base = today
                    week_start = base - timedelta(days=base.weekday())
                    if rel == 'last':
                        week_start = week_start - timedelta(days=7)
                    elif rel == 'next':
                        week_start = week_start + timedelta(days=7)
                    if wd_ta in tamil_week:
                        target_date = week_start + timedelta(days=tamil_week[wd_ta])
                if target_date:
                    iso = target_date.isoformat()
                    entities['date'] = iso
                    entities['date_range_start'] = iso
                    entities['date_range_end'] = iso

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

        # Normalize times to 24h HH:MM and promote single time
        entities = self._normalize_time_entities(entities)
        return entities

    @staticmethod
    def _to_24h(t_ampm: str) -> str:
        """Convert times like '9am','09am','12pm','12am','10pm' to HH:MM 24h."""
        m = re.match(r"^(\d{1,2})(am|pm)$", t_ampm)
        if not m:
            return t_ampm
        h = int(m.group(1))
        ap = m.group(2)
        if ap == 'am':
            if h == 12:
                h = 0
        else:  # pm
            if h != 12:
                h += 12
        return f"{h:02d}:00"

    def _normalize_time_entities(self, entities: dict):
        # If a combined 'time' only, promote to start_time for consistency
        if 'time' in entities and 'start_time' not in entities and 'end_time' not in entities:
            entities['start_time'] = entities.pop('time')
        # Normalize start/end
        for key in ['start_time','end_time']:
            if key in entities:
                entities[key] = self._to_24h(entities[key])
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
