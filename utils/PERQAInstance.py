class PERQAInstance:

    def __init__(self, name, session, raw_id, session_f, qas_f):
        self.name = name
        self.session = session
        self.raw_id = raw_id
        self.session_f = session_f
        self.qas_f = qas_f

    def __repr__(self):
        return "PERQA Instance: name=" + str(self.name) + ", id=" + str(self.raw_id)
