import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import os
from typing import List, Dict, Optional, Union

SINGLE_TIME_FALLBACK_MODE = os.getenv("SINGLE_TIME_FALLBACK_MODE", "plus1h").lower()
# Modes:
#  plus1h   -> end_time = start_time + 1 hour (capped same date 23:59)
#  now      -> end_time = current time (but if date in past AND now < start_time, still use +1h to avoid inversion)
#  partofday-> if part_of_day detected, end_time = min(part window end, start+1h) else behave like plus1h

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
        # Inference (every-request) self-learning (enabled by default; can cause drift)
        # Control via env NLP_INFER_LEARN (true/false) and NLP_INFER_RETRAIN_EVERY (default 10)
        self.inference_learn_enabled = str(os.getenv("NLP_INFER_LEARN", "true")).lower() in {"1","true","yes","on"}
        self.inference_retrain_every = int(os.getenv("NLP_INFER_RETRAIN_EVERY", "10") or 10)
        self._inference_added = 0

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
        # Heuristic override: patterns implying playback
        # Accept variants: "play from", "playback from", "play back from", "show playback from", "show play back from"
        lc = command_text.lower()
        if intent != 'playback':
            playback_range_pattern = r"(?:(?:show\s+)?play\s*back|(?:show\s+)?playback|play)\s+from\s+\d{1,2}(?::\d{2})?\s*(?:a\.m\.|p\.m\.|am|pm)\s+to\s+\d{1,2}(?::\d{2})?\s*(?:a\.m\.|p\.m\.|am|pm)"
            if re.search(playback_range_pattern, lc):
                intent = 'playback'
        # Fallback heuristic: if we clearly extracted a start & end time but classifier chose something else, treat as playback
        if intent != 'playback' and entities.get('start_time') and entities.get('end_time'):
            intent = 'playback'
        # Attempt auto-learning if enabled
        self._maybe_auto_learn(command_text, intent, score, entities)
        # Optional inference learning (append every request regardless of confidence)
        self.learn_from_inference(command_text, intent, entities, score)
        return {"intent": intent, "score": score, "entities": entities}

    # --------------------------------------------------------------
    # Inference self-learning (append each prediction to user dataset)
    # --------------------------------------------------------------
    def learn_from_inference(self, text: str, intent: str, entities: Dict[str,str], score: float) -> bool:
        """Persist this inference as a training example if enabled.

        WARNING: This uses model-predicted intent labels and can reinforce errors.
        Use only when you have very limited base data or are prototyping.
        Controlled by env NLP_INFER_LEARN=true. Retrain frequency controlled by
        NLP_INFER_RETRAIN_EVERY (default 25). Deduplicates on text string.
        """
        if not self.inference_learn_enabled:
            return False
        if not text or not intent:
            return False
        # Deduplicate exact text (case-insensitive)
        if any(t.lower() == text.lower() for t in self.texts):
            return False
        # Serialize entities with score
        ent_with_score = entities.copy() if entities else {}
        ent_with_score['pred_score'] = f"{score:.4f}"
        serialized = self._serialize_entities(ent_with_score)
        try:
            self._append_user_example(text, intent, serialized)
            self._inference_added += 1
            # Periodic retrain
            if self._inference_added % max(1, self.inference_retrain_every) == 0:
                self.load_data()
            return True
        except Exception:
            return False

    def extract_entities(self, command_text):
        text_lc = command_text.lower()
        # Part-of-day detection for inference of ambiguous times
        part_of_day = None
        for pod, variants in {
            'morning': ['morning','முற்பகல்'],
            'afternoon': ['afternoon','பிற்பகல்'],
            'evening': ['evening','சாயங்கால'],
            'night': ['night','இரவு']
        }.items():
            if any(v in text_lc for v in variants):
                part_of_day = pod
                break
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
        if part_of_day and 'part_of_day' not in entities:
            entities['part_of_day'] = part_of_day
        # Pattern: from 9 am to 10 am / 9am to 10am / 9 am - 10 am
        time_token = r"\d{1,2}(?::\d{2})?\s*(?:a\.m\.|p\.m\.|am|pm)"
        range_match = re.search(rf"from\s+({time_token})\s+to\s+({time_token})", text_lc)
        if not range_match:
            range_match = re.search(rf"({time_token})\s*(?:-|to)\s*({time_token})", text_lc)
        if range_match:
            entities['start_time'] = re.sub(r"\s+", "", range_match.group(1))
            entities['end_time'] = re.sub(r"\s+", "", range_match.group(2))

        # O'clock pattern (e.g., 10 o'clock / 10 o clock / 10 o’ clock)
        if 'start_time' not in entities:
            oclock_match = re.search(r"\b(\d{1,2})\s*(?:o['’]?\s*clock|o\s*clock|o'clock)\b", text_lc)
            if oclock_match:
                hour = int(oclock_match.group(1))
                # Infer am/pm from part_of_day if present
                if part_of_day == 'morning' and 1 <= hour <= 11:
                    entities['start_time'] = f"{hour}:00am"
                elif part_of_day in {'afternoon','evening','night'}:
                    # Treat 1-11 as pm, leave 12 as 12pm
                    if hour == 12:
                        entities['start_time'] = "12:00pm"
                    else:
                        entities['start_time'] = f"{hour}:00pm"
                else:
                    # Ambiguous: store as plain HH:00 (will be promoted & left as-is if cannot am/pm-normalize)
                    entities['start_time'] = f"{hour:02d}:00"

        # Single times (collect if not already captured)
        all_times = re.findall(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.m\.|p\.m\.|am|pm)\b", text_lc)
        if all_times and 'start_time' not in entities:
            if len(all_times) == 1:
                h, m, ap = all_times[0]
                minute = m or '00'
                entities['time'] = f"{h}:{minute}{ap.replace('.', '')}"
            else:
                h1,m1,ap1 = all_times[0]
                h2,m2,ap2 = all_times[-1]
                entities['start_time'] = f"{h1}:{m1 or '00'}{ap1}"
                entities['end_time'] = f"{h2}:{m2 or '00'}{ap2}"

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
        """Convert times like '9am','9:15am','09 a.m.','12pm','12:30 p.m.' to HH:MM 24h."""
        if not t_ampm:
            return t_ampm
        t = t_ampm.lower().replace('.', '')  # remove dots in a.m./p.m.
        t = re.sub(r"\s+", "", t)
        # Patterns: 9am, 9:15am, 09am, 09:05pm
        m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)$", t)
        if not m:
            return t_ampm  # return original if no match (already normalized?)
        h = int(m.group(1)); minute = m.group(2) or '00'; ap = m.group(3)
        if ap == 'am':
            if h == 12: h = 0
        else:
            if h != 12: h += 12
        return f"{h:02d}:{minute}"

    def _normalize_time_entities(self, entities: dict):
        # If a combined 'time' only, promote to start_time for consistency
        if 'time' in entities and 'start_time' not in entities and 'end_time' not in entities:
            entities['start_time'] = entities.pop('time')
        for key in ['start_time','end_time']:
            if key in entities:
                entities[key] = self._to_24h(entities[key])
        # Enhanced fallback
        if 'start_time' in entities and 'end_time' not in entities:
            from datetime import datetime as _dt, timedelta as _td, date as _date
            start_raw = entities['start_time']
            # Parse HH:MM or HH format
            m = re.match(r"^(\d{1,2})(?::(\d{2}))?$", start_raw)
            if m:
                sh = int(m.group(1)); sm = int(m.group(2) or '00')
            else:
                # If cannot parse, just set end = start (device may reject but we tried)
                entities['end_time'] = start_raw
                entities['fallback_strategy'] = 'unparsed_echo'
                return entities
            part_of_day = entities.get('part_of_day')
            # Determine date (if provided) to decide whether it's past
            today = _date.today()
            date_str = entities.get('date')
            date_obj = None
            if date_str:
                try:
                    date_obj = _dt.strptime(date_str, '%Y-%m-%d').date()
                except Exception:
                    date_obj = None
            # Helper to format time
            def fmt(h,mi):
                return f"{h:02d}:{mi:02d}"
            # Compute default +1h
            plus1h_h = sh
            plus1h_m = sm + 0
            # add 60 minutes
            plus1_total = sh*60 + sm + 60
            plus1h_h = plus1_total // 60
            plus1h_m = plus1_total % 60
            # Cap to 23:59 same date
            if plus1h_h > 23:
                plus1h_h, plus1h_m = 23, 59
            # Part-of-day windows
            pod_windows = {
                'morning': (6*60, 12*60),        # 06:00-12:00
                'afternoon': (12*60, 17*60),      # 12:00-17:00
                'evening': (17*60, 21*60),        # 17:00-21:00
                'night': (21*60, 24*60-1)         # 21:00-23:59
            }
            if SINGLE_TIME_FALLBACK_MODE == 'partofday' and part_of_day in pod_windows:
                start_minutes = sh*60 + sm
                window_start, window_end = pod_windows[part_of_day]
                # If user gave time outside window, keep +1h logic
                if window_start <= start_minutes <= window_end:
                    end_candidate = min(window_end, start_minutes + 60)
                    eh = end_candidate // 60; em = end_candidate % 60
                    entities['end_time'] = fmt(eh, em)
                    entities['fallback_strategy'] = 'partofday_window'
                else:
                    entities['end_time'] = fmt(plus1h_h, plus1h_m)
                    entities['fallback_strategy'] = 'plus1h_outside_window'
            elif SINGLE_TIME_FALLBACK_MODE == 'now':
                now = _dt.now()
                now_h = now.hour; now_m = now.minute
                # If date is past or (same date but now earlier than start) -> avoid inversion using +1h
                if (date_obj and date_obj < today) or (date_obj == today and (now_h*60 + now_m) < (sh*60 + sm)):
                    entities['end_time'] = fmt(plus1h_h, plus1h_m)
                    entities['fallback_strategy'] = 'plus1h_guard_for_past_or_inversion'
                else:
                    entities['end_time'] = fmt(now_h, now_m)
                    entities['fallback_strategy'] = 'now'
            else:  # plus1h default
                # If date in future AND plus1h crosses midnight, we still cap same day
                entities['end_time'] = fmt(plus1h_h, plus1h_m)
                entities['fallback_strategy'] = 'plus1h'
            return entities
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
