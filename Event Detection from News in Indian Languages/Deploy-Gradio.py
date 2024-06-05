import gradio as gr

def event_detection_and_classification(text):
    events = detect_events(text)
    event_types = [classify_event(event) for event in events]
    return {'events': events, 'event_types': event_types}

iface = gr.Interface(fn=event_detection_and_classification, inputs='text', outputs=['json'])
iface.launch()
