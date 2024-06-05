import spacy
nlp = spacy.load('en_core_web_sm')

def detect_events(text):
    doc = nlp(text)
    events = [ent.text for ent in doc.ents if ent.label_ in ['EVENT', 'DISASTER']]
    return events

news_df['events'] = news_df['content'].apply(detect_events)

def classify_event(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return 'natural' if prediction == 0 else 'man-made'

news_df['event_type'] = news_df['events'].apply(lambda events: [classify_event(event) for event in events])
news_df.to_csv('classified_news_data.csv', index=False)
