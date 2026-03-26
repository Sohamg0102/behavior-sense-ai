from collections import Counter


def interpret_behavior(emotion_history, emotion_labels):
    """
    Convert emotion history into behavioral insight
    """

    if len(emotion_history) == 0:
        return "No data"

    # Count frequency of emotions
    count = Counter(emotion_history)

    most_common = count.most_common(1)[0][0]
    most_common_emotion = emotion_labels[most_common]

    # Calculate variability
    unique_emotions = len(count)

    # 🔥 Rule-based interpretation

    # Case 1: Stable neutral
    if most_common_emotion == "neutral" and unique_emotions <= 2:
        return "User appears calm and focused"

    # Case 2: Mostly happy
    elif most_common_emotion == "happy":
        return "User appears engaged and positive"

    # Case 3: Presence of negative emotions
    elif "sad" in [emotion_labels[i] for i in emotion_history] or \
         "anger" in [emotion_labels[i] for i in emotion_history]:
        return "User may be experiencing stress or discomfort"

    # Case 4: High variation
    elif unique_emotions > 3:
        return "User shows fluctuating emotions (possible uncertainty)"

    # Default
    return "Behavior unclear but stable"