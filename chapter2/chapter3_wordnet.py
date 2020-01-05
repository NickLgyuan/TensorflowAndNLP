from nltk.corpus import wordnet as wn
word = 'car'
car_syns = wn.synsets(word)

syns_defs = [car_syns[i].definition() for i in range(len(car_syns))]

car_lemmas = car_syns[0].lemmas()[:3]

syn = car_syns[0]
print('\t',syn.hypernyms()[0].name(),'\n')

syn = car_syns[0]
print('\t',[hypo.name() for hypo in syn.hyponyms()[:3]],'\n')

syn = car_syns[2]
print('\t',[holo.name() for holo in syn.part_holonyms()],'\n')

syn = car_syns[0]
print('\t',[mero.name() for mero in syn.part_meronyms()[:3]],'\n')