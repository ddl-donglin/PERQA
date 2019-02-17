class PERQAInstance:
    """
    xfduan:     736
    dldi:       804
    lmzhang:    802
    zyzhao:     920
    zqzhu:      804
    chzhu:      702
    jwhu:       729
    xichen:     706
    """

    def __init__(self, name, session, raw_id, session_f, qas_f):
        self.name = name
        self.session = session
        self.raw_id = raw_id
        self.session_f = session_f
        self.qas_f = qas_f

    def __repr__(self):
        return "PERQA Instance: name=" + str(self.name) + ", id=" + str(self.raw_id)

    def get_qa_id(self):
        return self.name + '_' + str(self.raw_id)
