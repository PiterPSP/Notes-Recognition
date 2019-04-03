import numpy as np
def isFilled(noteCenter, threshold):
    (height, width) = np.shape(noteCenter)
    fill = 1 - (sum(sum(noteCenter))/(width * height))
    if fill > threshold:
        return True
    return False

def classifyNote(value, noteImage):
    notes = {
        0: "D",
        0.5: "E",
        1: "F",
        1.5: "G",
        2: "A",
        2.5: "B",
        3: "C",
        3.5: "D",
        4: "E",
        4.5: "F",
        5: "G",
        5.5: "A"
    }
    value = notes.get(value, "Err")
    (height, width) = np.shape(noteImage)
    name = "toDo"
    if height/width > 1.3 or height > 30:
        # staffed
        rightEdge = noteImage[:, int(width/5)*4:]
        if isFilled(rightEdge, 0.6):
            tail = False
            noteCenter = noteImage[int(height/12)*9 : height - int(height/12), int(width/4) : width - int(width/4)]
        else:
            tail = True
            noteCenter = noteImage[int(height/12)*9 : height - int(height/12), int(width/8) : int(width/8) * 3]
        if isFilled(noteCenter, 0.5):
            filled = True
        else:
            filled = False
        if filled:
            if tail:
                name = "1/8"
            else:
                name = "1/4"
        elif tail:
            name = "1/8e"
        else:
            name = "1/2"
    else:
        # not staffed
        noteCenter = noteImage[int(height/4) : height - int(height/4), int(width/4) : width - int(width/4)]
        if not isFilled(noteCenter, 0.5):
            name = "1e"
        else:
            name = "1"
    val = value
    return name, val
