

class CodeVocab(object):
    END_CODE = '<end>'
    PAD_CODE = '<pad>'
    UNK_CODE = '<unk>'

    def __init__(self):
        super().__init__()
        special_codes = [CodeVocab.END_CODE, CodeVocab.PAD_CODE, CodeVocab.UNK_CODE]

        self.special_codes = special_codes

        self.code2id = {}
        self.id2code = {}


        if self.special_codes is not None:
            self.add_code_list(self.special_codes)

    def add_code_list(self, code_list, rebuild=True):
        for code in code_list:
            if code not in self.code2id:
                self.code2id[code] = len(self.code2id)

        if rebuild:
            self._rebuild_id2code()

    def add_patients_visits(self, patients_visits):
        for patient in patients_visits:
            for visit in patient:
                self.add_code_list(visit)

        self._rebuild_id2code()

    def _rebuild_id2code(self):
        self.id2code = {i: t for t, i in self.code2id.items()}

    def get(self, item, default=None):
        return self.code2id.get(item, default)

    def __getitem__(self, item):
        return self.code2id[item]

    def __contains__(self, item):
        return item in self.code2id

    def __len__(self):
        return len(self.code2id)

    def __str__(self):
        return f'{len(self)} codes'
