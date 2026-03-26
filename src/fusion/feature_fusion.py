def fuse_features(emotion_history, emotion_labels, audio_state):
    """
    Combines emotion + behavior + audio into a final insight
    """

    if len(emotion_history) == 0:
        return "No data"

    # Get dominant emotion
    dominant_emotion_idx = max(set(emotion_history), key=emotion_history.count)
    dominant_emotion = emotion_labels[dominant_emotion_idx]

    # ----------- RULE-BASED FUSION ------------ #

    # Case 1: Neutral + Silent
    if dominant_emotion == "neutral" and "Silent" in audio_state:
        return "User is attentive but not verbally engaged"

    # Case 2: Neutral + Speaking
    if dominant_emotion == "neutral" and "Speaking" in audio_state:
        return "User is explaining or thinking aloud"

    # Case 3: Happy + Speaking
    if dominant_emotion == "happy" and "Speaking" in audio_state:
        return "User is actively engaged and positive"

    # Case 4: Sad + Silent
    if dominant_emotion == "sad" and "Silent" in audio_state:
        return "User may be disengaged or low energy"

    # Case 5: Angry + Speaking
    if dominant_emotion == "angry" and "Speaking" in audio_state:
        return "User might be frustrated or stressed"

    # Default fallback
    return f"User shows {dominant_emotion} behavior with {audio_state}"