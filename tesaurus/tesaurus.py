import json

class Tesaurus:
    def __init__(self, filename):
        # Membaca dictionary dari json
        self.mydict = self.load(filename)

    def load(self, filename):
        with open(filename) as data_file:
            data = json.load(data_file)
        return data

    # Mencari sinonim suatu kata
    def getSinonim(self, word):
        if word in self.mydict.keys():
            return self.mydict[word]['sinonim']
        else:
            return []

    # Mencari antonim suatu kata
    def getAntonim(self, word):
        if word in self.mydict.keys():
            if 'antonim' in self.mydict[word].keys():
                return self.mydict[word]['antonim']
        return []

# Contoh penggunaan
if __name__ == "__main__":
    tesaurus = Tesaurus(r'E:\IPEN\Semester 7\TA\Simulasi - Augment\tesaurus\dict.json')
    print(tesaurus.getSinonim('senang'))
    antonim_senang = tesaurus.getAntonim('senang')
    if antonim_senang:
        print(tesaurus.getSinonim(antonim_senang[0]))
    else:
        print("Tidak ada antonim untuk 'senang'")
    print(tesaurus.getSinonim('anna'))
