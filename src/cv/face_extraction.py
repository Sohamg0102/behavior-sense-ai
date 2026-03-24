def extract_faces(frame, detections, h, w):
    faces = []

    if detections:
        for detection in detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            face = frame[y:y + height, x:x + width]

            faces.append((face, (x, y, width, height)))

    return faces