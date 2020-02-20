class Recorder():
    def __init__(self, id, label, first_frame):
        self.id = id
        self.label = label
        self.frames = [first_frame]

    def update(self, frame):
        self.frames.append(frame)

    def num_frames(self):
        return len(self.frames)