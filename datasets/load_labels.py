def read_label_file(label_file):
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            data = line.strip().split(' ')
            label = {
                'type': data[0],
                'truncated': float(data[1]),
                'occluded': int(data[2]),
                'alpha': float(data[3]),
                'bbox': [float(x) for x in data[4:8]],
                'dimensions': [float(x) for x in data[8:11]],  # h, w, l
                'location': [float(x) for x in data[11:14]],
                'rotation_y': float(data[14])
            }
            labels.append(label)
    return labels
